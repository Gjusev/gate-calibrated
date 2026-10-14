"""Simulated human (plan section 4.8): a perfect oracle.

Approves an action iff it is correct per the gold-action labeling. Every
ask counts as one approval requested, regardless of the outcome.
"""


class Oracle:
    def __init__(self):
        self.approvals_requested = 0

    def approve(self, record) -> bool:
        self.approvals_requested += 1
        return bool(record.correct)
