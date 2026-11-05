"""CPU-only timing/join regressions. No weights, torch or model calls. [AI-CODE]"""
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import json

from segment_timing import validate_segments, audit_mapping


def declared_segments():
    return [{"start": start, "end": start + 149, "num_frames": 81, "keep_frames": 80}
            for start in (0, 150, 300)]


class TimingTests(unittest.TestCase):
    def test_full_duration_and_original_audit_indices(self):
        segments = validate_segments(declared_segments(), 0, 449, 30.0, 16)
        self.assertEqual([(s["output_start"], s["output_end"]) for s in segments],
                         [(0, 79), (80, 159), (160, 239)])
        self.assertEqual(sum(s["keep_frames"] for s in segments) / 16, 15)
        self.assertEqual([s["discard_tail_frames"] for s in segments], [1, 1, 1])
        self.assertEqual(audit_mapping(segments, [0, 45, 398, 390, 391], 30, 16, [390, 391]),
                         {0: 0, 45: 24, 398: 212, 390: 208, 391: 209})

    def test_no_gap_overlap_or_uncovered_tail(self):
        for index, key, value in [(1, "start", 151), (1, "start", 149), (2, "end", 448)]:
            with self.subTest(index=index, key=key, value=value):
                segments = declared_segments()
                segments[index][key] = value
                with self.assertRaises(ValueError):
                    validate_segments(segments, 0, 449, 30, 16)

    def test_boundary_and_counts_cannot_be_rounded_or_silently_trimmed(self):
        for items in [[{"start": 0, "end": 148, "num_frames": 81, "keep_frames": 80}],
                      [{"start": 0, "end": 149, "num_frames": 81, "keep_frames": 81}],
                      [{"start": 0, "end": 149, "num_frames": 85, "keep_frames": 80}]]:
            with self.assertRaises(ValueError):
                validate_segments(items, 0, items[-1]["end"], 30, 16)

    def test_pair_cannot_cross_seam_or_collapse_to_one_output(self):
        segments = validate_segments(declared_segments(), 0, 449, 30, 16)
        with self.assertRaisesRegex(ValueError, "within one"):
            audit_mapping(segments, [149, 150], 30, 16, [149, 150])
        with self.assertRaisesRegex(ValueError, "collide"):
            audit_mapping(segments, [391, 392], 30, 16, [391, 392])


class JoinTests(unittest.TestCase):
    def test_recover_only_logged_matching_completed_child(self):
        import segmented_infer as runner
        with tempfile.TemporaryDirectory(prefix="vace_recovery_test_") as temp:
            folder = Path(temp)
            output = folder / "segment_00" / "raw_video.mp4"
            output.parent.mkdir()
            output.write_bytes(b"CPU fixture only; validate_raw is isolated in this test")
            expected = {"seed": 2026, "segment": {"index": 0}, "audit_output_frames": {0: 0, 45: 24}}
            record = {"status": "complete", "plan": {**expected, "output_sha256": runner.core.digest(output)}}
            path = output.parent / "runs" / "test_invocation" / "invocation.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(record), encoding="utf-8")
            with patch.object(runner, "validate_raw"), patch.object(runner.core, "update_log") as update:
                with self.assertRaises(FileExistsError):
                    runner.recover_completed_child(output, expected, folder / "test_log.csv")
                row = {"notes": json.dumps({"invocation_id": "CPU_TEST_ONLY"})}
                record["log_row"] = row
                path.write_text(json.dumps(record), encoding="utf-8")
                self.assertEqual(runner.recover_completed_child(output, expected, folder / "test_log.csv"), path)
                update.assert_called_once_with(folder / "test_log.csv", "CPU_TEST_ONLY", row)
                with self.assertRaises(FileExistsError):
                    runner.recover_completed_child(output, {**expected, "seed": 7}, folder / "test_log.csv")
                output.write_bytes(b"changed bytes")
                with self.assertRaises(FileExistsError):
                    runner.recover_completed_child(output, expected, folder / "test_log.csv")

    def test_real_mp4_join_preserves_prefixes_and_discards_only_padding(self):
        # Actual FFmpeg encode/decode, deliberately obvious last padding frame.
        import cv2
        import numpy as np
        import imageio_ffmpeg
        from segmented_infer import join_segments
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        def decode(path):
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
            finally:
                cap.release()
            return np.asarray(frames), fps

        with tempfile.TemporaryDirectory(prefix="vace_timing_test_") as temp:
            folder = Path(temp)
            items, expected = [], []
            for i in range(3):
                pixels = np.empty((81, 32, 32, 3), dtype=np.uint8)
                for k in range(80):
                    pixels[k] = [30 + i * 65 + k % 16, 60 + i * 12, 90]
                pixels[80] = [255, 0, 255]
                path = folder / f"segment_{i}.mp4"
                subprocess.run([ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", "32x32", "-r", "16", "-i", "pipe:0", "-an",
                                "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(path)],
                               input=pixels.tobytes(), check=True, capture_output=True)
                raw, fps = decode(path)
                self.assertEqual((len(raw), fps), (81, 16))
                expected.extend(raw[:80])
                items.append({"path": path, "keep_frames": 80})
            joined = folder / "joined.mp4"
            join_segments(items, joined, 16, ffmpeg)
            actual, fps = decode(joined)
            self.assertEqual((len(actual), fps), (240, 16))
            np.testing.assert_array_equal(actual, np.asarray(expected))
            with self.assertRaises(FileExistsError):
                join_segments(items, joined, 16, ffmpeg)

    def test_failed_attempt_resumes_without_regenerating_completed_segments(self):
        import cv2
        import imageio_ffmpeg
        import segmented_infer as runner
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory(prefix="vace_resume_test_") as temp:
            folder = Path(temp)
            writer = folder / "fake_model_for_cpu_test.py"
            writer.write_text(
                "import sys\nfrom pathlib import Path\nimport cv2,numpy as np\n"
                "i=int(sys.argv[1]);out=Path(sys.argv[2]);counter=Path(sys.argv[3])\n"
                "n=int(counter.read_text())+1 if counter.exists() else 1\n"
                "counter.write_text(str(n))\n"
                "if i==1 and n==1:raise SystemExit(8)\n"
                "w=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),16,(32,32))\n"
                "for k in range(81):w.write(np.full((32,32,3),40+i*60,dtype=np.uint8))\n"
                "w.release()\n", encoding="utf-8")
            timing = validate_segments(declared_segments(), 0, 449, 30, 16)
            plans = [{"interval": {"start": s["start"], "end": s["end"]},
                      "segment": s, "num_frames": 81, "fps": 16, "width": 32, "height": 32}
                     for s in timing]
            work = folder / "segmented_attempt"
            outputs = [work / f"segment_{i:02d}" / "raw_video.mp4" for i in range(3)]
            args = SimpleNamespace(output=folder / "output.mp4", resume=False, fps=16, width=32, height=32)
            attempt = {"fingerprint": "CPU_TEST_ONLY", "segments": plans, "joined_frames": 240}

            def command(_args, index, output):
                import sys
                return [sys.executable, str(writer), str(index), str(output), str(folder / f"count_{index}.txt")]

            def probe(path):
                cap = cv2.VideoCapture(str(path))
                try:
                    return {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                            "fps": cap.get(cv2.CAP_PROP_FPS),
                            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
                finally:
                    cap.release()

            with patch.object(runner, "child_command", command), patch.object(runner.core, "probe", probe), \
                    patch.object(runner.shutil, "which", return_value=ffmpeg):
                with self.assertRaisesRegex(RuntimeError, "Segment 2 failed"):
                    runner.execute(args, attempt, outputs, work)
                first_hash = runner.core.digest(outputs[0])
                self.assertFalse(outputs[2].exists())
                args.resume = True
                runner.execute(args, attempt, outputs, work)
                self.assertEqual([int((folder / f"count_{i}.txt").read_text()) for i in range(3)], [1, 2, 1])
                self.assertEqual(runner.core.digest(outputs[0]), first_hash)
                self.assertEqual(probe(args.output)["frames"], 240)
                state = json.loads((work / "attempt.json").read_text())
                self.assertEqual(state["status"], "complete")
                # Repeating an already complete resume performs no new child invocation.
                runner.execute(args, attempt, outputs, work)
                self.assertEqual([int((folder / f"count_{i}.txt").read_text()) for i in range(3)], [1, 2, 1])
                with self.assertRaisesRegex(ValueError, "changed"):
                    runner.execute(args, {**attempt, "fingerprint": "CHANGED_INPUT"}, outputs, work)


if __name__ == "__main__":
    unittest.main(verbosity=2)
