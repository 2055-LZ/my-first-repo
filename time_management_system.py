from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


DAY_START = "06:00"
DAY_END = "23:30"
ENGLISH_ONLY_START = "06:30"
ENGLISH_ONLY_END = "08:30"
DEEP_MATH_START = "20:00"
DEEP_MATH_END = "22:00"


@dataclass
class Task:
    name: str
    category: str
    total_minutes: int
    splittable: bool = True
    interruptible: bool = True
    min_block: int = 25
    max_block: int = 35
    completed_minutes: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def remaining_minutes(self) -> int:
        return max(0, self.total_minutes - self.completed_minutes)


@dataclass
class FixedEvent:
    name: str
    start: datetime
    end: datetime
    is_class: bool = False


@dataclass
class Block:
    task_name: str
    start: datetime
    end: datetime
    reason: str
    low_reliability: bool = False

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)


@dataclass
class EnergyProfile:
    focus_windows: List[Tuple[str, str]] = field(default_factory=lambda: [("09:00", "11:30"), ("14:00", "17:00"), ("20:00", "22:00")])

    def suggest_block_size(self, task: Task) -> int:
        if not task.splittable:
            return task.remaining_minutes
        center = (task.min_block + task.max_block) // 2
        return max(task.min_block, min(center, task.max_block))


class TimeManagementSystem:
    def __init__(
        self,
        date: datetime,
        tasks: List[Task],
        fixed_events: List[FixedEvent],
        energy_profile: Optional[EnergyProfile] = None,
        allow_class_time: bool = False,
    ) -> None:
        self.date = date
        self.tasks = tasks
        self.fixed_events = sorted(fixed_events, key=lambda e: e.start)
        self.energy_profile = energy_profile or EnergyProfile()
        self.allow_class_time = allow_class_time
        self.blocks: List[Block] = []

    def schedule_day(self) -> List[Block]:
        self.blocks = []
        work_items = self._expand_tasks_by_shape_ratio(self.tasks)
        cursor = self._at(DAY_START)

        # 固定安排：高数（深度）20:00-22:00。
        deep_math_items = [t for t in work_items if t.category == "math_deep"]
        other_items = [t for t in work_items if t.category != "math_deep"]
        if deep_math_items:
            deep_task = deep_math_items[0]
            deep_start, deep_end = self._at(DEEP_MATH_START), self._at(DEEP_MATH_END)
            self._append_block(deep_task, deep_start, deep_end, "高数深度固定窗口")
            deep_task.completed_minutes += 120
            self._append_rest(deep_end, 20)

        while True:
            task = self._next_available_task(other_items)
            if not task:
                break

            slot = self._find_next_slot(cursor, task)
            if not slot:
                break
            start, end, low_rel = slot
            reason = "常规调度"
            if low_rel:
                reason = "低可靠：上课时段可中断插入"
            self._append_block(task, start, end, reason, low_rel)
            task.completed_minutes += int((end - start).total_seconds() // 60)
            cursor = end

        self.blocks.sort(key=lambda b: b.start)
        return self.blocks

    def rolling_reschedule(self, current_time: datetime, disruption_minutes: int) -> List[Block]:
        disruption_end = current_time + timedelta(minutes=disruption_minutes)
        past_blocks: List[Block] = []
        remaining_items: List[Task] = []

        for block in sorted(self.blocks, key=lambda b: b.start):
            if block.end <= current_time:
                past_blocks.append(block)
            elif block.start < current_time < block.end:
                done = Block(block.task_name, block.start, current_time, block.reason + "（已完成部分）", block.low_reliability)
                past_blocks.append(done)
                remaining_minutes = int((block.end - current_time).total_seconds() // 60)
                remaining_items.append(
                    self._carry_over_task_from_block(block, remaining_minutes)
                )
            else:
                remaining_minutes = int((block.end - block.start).total_seconds() // 60)
                remaining_items.append(self._carry_over_task_from_block(block, remaining_minutes))

        self.blocks = sorted(past_blocks, key=lambda b: b.start)
        cursor = disruption_end

        # 保持最小扰动：保留原有顺序重新塞回。
        for task in remaining_items:
            while task.remaining_minutes > 0:
                slot = self._find_next_slot(cursor, task)
                if not slot:
                    break
                start, end, low_rel = slot
                reason = "滚动重排（最小扰动）"
                if low_rel:
                    reason += " + 低可靠"
                self._append_block(task, start, end, reason, low_rel)
                task.completed_minutes += int((end - start).total_seconds() // 60)
                cursor = end

        self.blocks.sort(key=lambda b: b.start)
        return self.blocks

    def _expand_tasks_by_shape_ratio(self, tasks: List[Task]) -> List[Task]:
        expanded: List[Task] = []
        for task in tasks:
            if task.category != "math_non_deep":
                expanded.append(task)
                continue

            total = task.total_minutes
            practice_minutes = round(total * 0.6)
            error_minutes = total - practice_minutes
            expanded.extend(
                [
                    Task(
                        name=f"{task.name}-刷题",
                        category="math_practice",
                        total_minutes=practice_minutes,
                        splittable=True,
                        interruptible=True,
                        min_block=25,
                        max_block=35,
                        metadata={"ratio": "6"},
                    ),
                    Task(
                        name=f"{task.name}-错题整理",
                        category="math_error_review",
                        total_minutes=error_minutes,
                        splittable=True,
                        interruptible=True,
                        min_block=25,
                        max_block=35,
                        metadata={"ratio": "4"},
                    ),
                ]
            )
        return expanded

    def _next_available_task(self, tasks: List[Task]) -> Optional[Task]:
        for task in tasks:
            if task.remaining_minutes > 0:
                return task
        return None

    def _find_next_slot(self, cursor: datetime, task: Task) -> Optional[Tuple[datetime, datetime, bool]]:
        t = max(cursor, self._at(DAY_START))
        earliest_start = task.metadata.get("earliest_start")
        if earliest_start:
            t = max(t, datetime.fromisoformat(earliest_start))
        end_of_day = self._at(DAY_END)
        step = timedelta(minutes=5)

        while t < end_of_day:
            if self._inside_english_only_window(t) and task.category != "english":
                t = min(self._at(ENGLISH_ONLY_END), t + step)
                continue

            if task.category == "math_deep":
                deep_s, deep_e = self._at(DEEP_MATH_START), self._at(DEEP_MATH_END)
                if t <= deep_s:
                    return deep_s, deep_e, False
                return None

            block_len = self.energy_profile.suggest_block_size(task)
            if not task.splittable:
                block_len = task.remaining_minutes
            else:
                block_len = min(block_len, task.remaining_minutes)
            candidate_end = t + timedelta(minutes=block_len)

            if candidate_end > end_of_day:
                return None

            conflict, low_rel = self._has_conflict_or_low_reliability(t, candidate_end, task)
            if not conflict:
                return t, candidate_end, low_rel
            t += step

        return None

    def _has_conflict_or_low_reliability(self, start: datetime, end: datetime, task: Task) -> Tuple[bool, bool]:
        low_reliability = False

        for event in self.fixed_events:
            overlap = not (end <= event.start or start >= event.end)
            if not overlap:
                continue

            if event.is_class and self.allow_class_time and task.interruptible:
                low_reliability = True
                continue
            return True, False

        for block in self.blocks:
            overlap = not (end <= block.start or start >= block.end)
            if overlap:
                return True, False

        return False, low_reliability

    def _append_block(self, task: Task, start: datetime, end: datetime, reason: str, low_reliability: bool = False) -> None:
        self.blocks.append(Block(task.name, start, end, reason, low_reliability))

    def _carry_over_task_from_block(self, block: Block, minutes: int) -> Task:
        if block.task_name == "休息":
            return Task(
                name=block.task_name,
                category="rest",
                total_minutes=minutes,
                splittable=False,
                interruptible=False,
                min_block=minutes,
                max_block=minutes,
                metadata={"source": "rolling", "earliest_start": block.start.isoformat()},
            )
        if "高数（深度）" in block.task_name:
            return Task(
                name=block.task_name,
                category="math_deep",
                total_minutes=minutes,
                splittable=False,
                interruptible=False,
                min_block=120,
                max_block=120,
                metadata={"source": "rolling", "earliest_start": block.start.isoformat()},
            )
        return Task(
            name=block.task_name,
            category="carry_over",
            total_minutes=minutes,
            splittable=True,
            interruptible=True,
            min_block=25,
            max_block=35,
            metadata={"source": "rolling", "earliest_start": block.start.isoformat()},
        )

    def _append_rest(self, start: datetime, minutes: int) -> None:
        end = start + timedelta(minutes=minutes)
        self.blocks.append(Block("休息", start, end, "高数深度后恢复", False))

    def _inside_english_only_window(self, t: datetime) -> bool:
        return self._at(ENGLISH_ONLY_START) <= t < self._at(ENGLISH_ONLY_END)

    def _at(self, hhmm: str) -> datetime:
        hour, minute = map(int, hhmm.split(":"))
        return self.date.replace(hour=hour, minute=minute, second=0, microsecond=0)


def print_schedule(blocks: List[Block], title: str) -> None:
    print(f"\n=== {title} ===")
    for b in sorted(blocks, key=lambda x: x.start):
        tag = " [低可靠]" if b.low_reliability else ""
        print(f"{b.start.strftime('%H:%M')} - {b.end.strftime('%H:%M')} | {b.task_name}{tag} | {b.reason}")


def demo() -> None:
    base_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    tasks = [
        Task(name="英语听读", category="english", total_minutes=90, splittable=True, interruptible=True),
        Task(name="高数（深度）", category="math_deep", total_minutes=120, splittable=False, interruptible=False, min_block=120, max_block=120),
        Task(name="高数（非深度）", category="math_non_deep", total_minutes=150, splittable=True, interruptible=True),
        Task(name="课程复盘", category="review", total_minutes=60, splittable=True, interruptible=True),
    ]

    fixed_events = [
        FixedEvent("线性代数课", start=base_day.replace(hour=9, minute=0), end=base_day.replace(hour=10, minute=30), is_class=True),
        FixedEvent("午餐", start=base_day.replace(hour=12, minute=0), end=base_day.replace(hour=12, minute=40), is_class=False),
    ]

    system = TimeManagementSystem(
        date=base_day,
        tasks=tasks,
        fixed_events=fixed_events,
        energy_profile=EnergyProfile(),
        allow_class_time=True,
    )

    original = system.schedule_day()
    print_schedule(original, "初始排程")

    current_time = base_day.replace(hour=15, minute=10)
    updated = system.rolling_reschedule(current_time=current_time, disruption_minutes=45)
    print_schedule(updated, "滚动重排后")


if __name__ == "__main__":
    demo()
