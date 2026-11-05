"""Launch human audit annotation on the retimed poster_03 source. [AI-CODE]

No audit coordinates or masks are generated. The user clicks/draws in the
supplied GUI; the existing launcher resumes missing frames and checks them.
"""
from pathlib import Path

from annotate_poster_01 import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "personal_preparation/poster_03/audit_plan.json")
