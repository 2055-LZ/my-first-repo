"""
可迭代精密时间管理系统（CLI版）

使用方式：
1) 初始排程（非交互）
   python time_management_system.py plan

2) 滚动重排（非交互）
   python time_management_system.py roll --time 14:15 --disruption 0

3) 交互模式（未提供参数时自动询问）
   python time_management_system.py roll
   python time_management_system.py plan

4) 使用 JSON 配置覆盖默认任务/固定事件
   python time_management_system.py plan --config config.json

config.json 示例：
{
  "allow_class_time": true,
  "tasks": [
    {"name": "英语听读", "category": "english", "total_minutes": 120},
    {"name": "高数（深度）", "category": "math_deep", "total_minutes": 120, "splittable": false, "interruptible": false, "min_block": 120, "max_block": 120},
    {"name": "高数（非深度）", "category": "math_non_deep", "total_minutes": 180},
    {"name": "课程复盘", "category": "review", "total_minutes": 60}
  ],
  "fixed_events": [
    {"name": "早餐", "start": "06:00", "end": "06:30", "is_class": false},
    {"name": "上午上课", "start": "08:30", "end": "12:10", "is_class": true}
  ]
}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DAY_START = "05:30"
DAY_END = "23:30"
ENGLISH_ONLY_START = "06:30"
ENGLISH_ONLY_END = "08:30"
DEEP_MATH_START = "20:00"
DEEP_MATH_END = "22:00"
MIN_SPLIT_BLOCK = 25


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
    focus_windows: List[Tuple[str, str]] = field(default_factory=lambda: [("06:30", "08:30"), ("12:40", "13:50"), ("20:00", "22:00")])

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

        deep_math_items = [t for t in work_items if t.category == "math_deep"]
        other_items = [t for t in work_items if t.category != "math_deep"]
        if deep_math_items:
            deep_task = deep_math_items[0]
            deep_start, deep_end = self._at(DEEP_MATH_START), self._at(DEEP_MATH_END)
            self._append_block(deep_task, deep_start, deep_end, "高数深度固定窗口")
            deep_task.completed_minutes += min(120, deep_task.total_minutes)
            self._append_rest(deep_end, 20)

        while True:
            task = self._next_available_task(other_items, cursor)
            if not task:
                break

            slot = self._find_next_slot(cursor, task)
            if not slot:
                task.metadata["blocked"] = "1"
                continue

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
                remaining_items.append(self._carry_over_task_from_block(block, remaining_minutes))
            else:
                remaining_minutes = int((block.end - block.start).total_seconds() // 60)
                remaining_items.append(self._carry_over_task_from_block(block, remaining_minutes))

        self.blocks = sorted(past_blocks, key=lambda b: b.start)
        cursor = disruption_end

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
                    Task(name=f"{task.name}-刷题", category="math_practice", total_minutes=practice_minutes, splittable=True, interruptible=True, min_block=25, max_block=35, metadata={"ratio": "6"}),
                    Task(name=f"{task.name}-错题整理", category="math_error_review", total_minutes=error_minutes, splittable=True, interruptible=True, min_block=25, max_block=35, metadata={"ratio": "4"}),
                ]
            )
        return expanded

    def _next_available_task(self, tasks: List[Task], cursor: datetime) -> Optional[Task]:
        english_task = self._english_task_with_remaining(tasks)
        if english_task and cursor < self._at(ENGLISH_ONLY_END):
            return english_task

        for task in tasks:
            if task.metadata.get("blocked") == "1" or task.remaining_minutes <= 0:
                continue
            if task.splittable and task.remaining_minutes < MIN_SPLIT_BLOCK:
                continue
            return task
        return None

    def _english_task_with_remaining(self, tasks: List[Task]) -> Optional[Task]:
        for task in tasks:
            if task.category == "english" and task.remaining_minutes > 0:
                return task
        return None

    def _find_next_slot(self, cursor: datetime, task: Task) -> Optional[Tuple[datetime, datetime, bool]]:
        t = max(cursor, self._at(DAY_START))
        earliest_start = task.metadata.get("earliest_start")
        if earliest_start:
            t = max(t, datetime.fromisoformat(earliest_start))
        if task.category == "english" and t < self._at(ENGLISH_ONLY_START):
            t = self._at(ENGLISH_ONLY_START)

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

            block_len = self._calculate_block_len(task)
            if block_len is None:
                return None
            candidate_end = t + timedelta(minutes=block_len)
            if candidate_end > end_of_day:
                return None

            conflict, low_rel = self._has_conflict_or_low_reliability(t, candidate_end, task)
            if not conflict:
                return t, candidate_end, low_rel
            t += step
        return None

    def _calculate_block_len(self, task: Task) -> Optional[int]:
        if not task.splittable:
            return task.remaining_minutes
        if task.remaining_minutes < MIN_SPLIT_BLOCK:
            return None
        preferred = self.energy_profile.suggest_block_size(task)
        block_len = min(preferred, task.remaining_minutes)
        if block_len < MIN_SPLIT_BLOCK:
            return None
        return block_len

    def _has_conflict_or_low_reliability(self, start: datetime, end: datetime, task: Task) -> Tuple[bool, bool]:
        low_reliability = False
        for event in self.fixed_events:
            overlap = self._event_overlap(start, end, event)
            if not overlap:
                continue
            if event.is_class and self.allow_class_time and task.interruptible:
                low_reliability = True
                continue
            return True, False

        for block in self.blocks:
            if not (end <= block.start or start >= block.end):
                return True, False
        return False, low_reliability

    def _event_overlap(self, start: datetime, end: datetime, event: FixedEvent) -> bool:
        if not event.is_class:
            return not (end <= event.start or start >= event.end)
        return not (end <= event.start or start > event.end)

    def _append_block(self, task: Task, start: datetime, end: datetime, reason: str, low_reliability: bool = False) -> None:
        self.blocks.append(Block(task.name, start, end, reason, low_reliability))

    def _carry_over_task_from_block(self, block: Block, minutes: int) -> Task:
        if block.task_name == "休息":
            return Task(block.task_name, "rest", minutes, False, False, minutes, minutes, metadata={"source": "rolling", "earliest_start": block.start.isoformat()})
        if "高数（深度）" in block.task_name:
            return Task(block.task_name, "math_deep", minutes, False, False, 120, 120, metadata={"source": "rolling", "earliest_start": block.start.isoformat()})
        return Task(block.task_name, "carry_over", minutes, True, True, 25, 35, metadata={"source": "rolling", "earliest_start": block.start.isoformat()})

    def _append_rest(self, start: datetime, minutes: int) -> None:
        self.blocks.append(Block("休息", start, start + timedelta(minutes=minutes), "高数深度后恢复", False))

    def _inside_english_only_window(self, t: datetime) -> bool:
        return self._at(ENGLISH_ONLY_START) <= t < self._at(ENGLISH_ONLY_END)

    def _at(self, hhmm: str) -> datetime:
        hour, minute = map(int, hhmm.split(":"))
        return self.date.replace(hour=hour, minute=minute, second=0, microsecond=0)


def print_schedule(blocks: List[Block], title: str) -> None:
    print(f"\n=== {title}（简表）===")
    for b in sorted(blocks, key=lambda x: x.start):
        tag = " [低可靠]" if b.low_reliability else ""
        print(f"{b.start.strftime('%H:%M')} - {b.end.strftime('%H:%M')} | {b.task_name}{tag} | {b.reason}")


def print_timeline(day_start: datetime, day_end: datetime, fixed_events: List[FixedEvent], blocks: List[Block]) -> None:
    points = {day_start, day_end}
    for event in fixed_events:
        if event.end > day_start and event.start < day_end:
            points.add(max(day_start, event.start))
            points.add(min(day_end, event.end))
    for block in blocks:
        if block.end > day_start and block.start < day_end:
            points.add(max(day_start, block.start))
            points.add(min(day_end, block.end))

    print("\n=== 完整时间线 ===")
    timeline = sorted(points)
    for i in range(len(timeline) - 1):
        seg_start, seg_end = timeline[i], timeline[i + 1]
        covering_events = [e for e in fixed_events if not (seg_end <= e.start or seg_start >= e.end)]
        covering_blocks = [b for b in blocks if not (seg_end <= b.start or seg_start >= b.end)]

        if covering_blocks and covering_events:
            task = covering_blocks[0]
            tag = " [低可靠]" if task.low_reliability else ""
            label = f"任务:{task.task_name}{tag} | 固定事件:{','.join(e.name for e in covering_events)}"
        elif covering_blocks:
            task = covering_blocks[0]
            tag = " [低可靠]" if task.low_reliability else ""
            label = f"任务:{task.task_name}{tag}"
        elif covering_events:
            label = "固定事件:" + ",".join(e.name for e in covering_events)
        else:
            label = "空闲/机动"
        print(f"{seg_start.strftime('%H:%M')} - {seg_end.strftime('%H:%M')} | {label}")


def validate_schedule_constraints(blocks: List[Block], fixed_events: List[FixedEvent], english_target_minutes: int) -> None:
    breakfast = next((e for e in fixed_events if e.name == "早餐"), None)
    breakfast_ok = True
    if breakfast:
        breakfast_ok = all(block.end <= breakfast.start or block.start >= breakfast.end for block in blocks)

    english_window_start = datetime.strptime(ENGLISH_ONLY_START, "%H:%M").time()
    english_window_end = datetime.strptime(ENGLISH_ONLY_END, "%H:%M").time()
    english_minutes_in_window = 0
    class_low_rel_ok = True

    for block in blocks:
        if english_window_start <= block.start.time() < english_window_end and "英语" in block.task_name:
            english_minutes_in_window += block.duration_minutes

        for event in fixed_events:
            if not event.is_class:
                continue
            class_overlap = not (block.end <= event.start or block.start > event.end)
            if class_overlap and (not block.low_reliability or "低可靠" not in block.reason):
                class_low_rel_ok = False

    english_ok = english_minutes_in_window >= min(english_target_minutes, 120)

    print("\n=== 约束验证 ===")
    print(f"早餐窗口无任务: {'PASS' if breakfast_ok else 'FAIL'}")
    print(f"英语窗口优先填充: {'PASS' if english_ok else 'FAIL'} (窗口英语分钟={english_minutes_in_window})")
    print(f"上课窗口低可靠一致标记: {'PASS' if class_low_rel_ok else 'FAIL'}")


def build_default_fixed_events(base_day: datetime) -> List[FixedEvent]:
    return [
        FixedEvent("早餐", start=base_day.replace(hour=6, minute=0), end=base_day.replace(hour=6, minute=30), is_class=False),
        FixedEvent("上午上课", start=base_day.replace(hour=8, minute=30), end=base_day.replace(hour=12, minute=10), is_class=True),
        FixedEvent("午饭", start=base_day.replace(hour=12, minute=10), end=base_day.replace(hour=12, minute=40), is_class=False),
        FixedEvent("下午上课", start=base_day.replace(hour=14, minute=0), end=base_day.replace(hour=17, minute=40), is_class=True),
        FixedEvent("健身", start=base_day.replace(hour=18, minute=0), end=base_day.replace(hour=19, minute=0), is_class=False),
        FixedEvent("晚饭", start=base_day.replace(hour=19, minute=0), end=base_day.replace(hour=19, minute=40), is_class=False),
    ]


def build_tasks(english_minutes: int, math_deep_minutes: int, math_non_deep_minutes: int, review_minutes: int) -> List[Task]:
    return [
        Task(name="英语听读", category="english", total_minutes=english_minutes, splittable=True, interruptible=True),
        Task(name="高数（深度）", category="math_deep", total_minutes=math_deep_minutes, splittable=False, interruptible=False, min_block=120, max_block=120),
        Task(name="高数（非深度）", category="math_non_deep", total_minutes=math_non_deep_minutes, splittable=True, interruptible=True),
        Task(name="课程复盘", category="review", total_minutes=review_minutes, splittable=True, interruptible=True),
    ]


def parse_date(date_str: Optional[str]) -> datetime:
    if not date_str:
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime.strptime(date_str, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)


def parse_time_on_date(base_day: datetime, hhmm: str) -> datetime:
    hour, minute = map(int, hhmm.split(":"))
    return base_day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def ask(prompt: str, default: Optional[str] = None) -> str:
    tip = f" [{default}]" if default is not None else ""
    try:
        val = input(f"{prompt}{tip}: ").strip()
    except EOFError:
        return default or ""
    return val if val else (default or "")


def load_config(path: Path, base_day: datetime) -> Tuple[Optional[bool], Optional[List[Task]], Optional[List[FixedEvent]]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    allow_class_time = data.get("allow_class_time")

    tasks_data = data.get("tasks")
    tasks = None
    if tasks_data:
        tasks = []
        for t in tasks_data:
            tasks.append(
                Task(
                    name=t["name"],
                    category=t["category"],
                    total_minutes=int(t["total_minutes"]),
                    splittable=bool(t.get("splittable", True)),
                    interruptible=bool(t.get("interruptible", True)),
                    min_block=int(t.get("min_block", 25)),
                    max_block=int(t.get("max_block", 35)),
                )
            )

    fixed_data = data.get("fixed_events")
    fixed_events = None
    if fixed_data:
        fixed_events = []
        for e in fixed_data:
            fixed_events.append(
                FixedEvent(
                    name=e["name"],
                    start=parse_time_on_date(base_day, e["start"]),
                    end=parse_time_on_date(base_day, e["end"]),
                    is_class=bool(e.get("is_class", False)),
                )
            )

    return allow_class_time, tasks, fixed_events


def run(mode: str, base_day: datetime, allow_class_time: bool, tasks: List[Task], fixed_events: List[FixedEvent], current_time: Optional[str], disruption_minutes: Optional[int]) -> None:
    system = TimeManagementSystem(
        date=base_day,
        tasks=tasks,
        fixed_events=fixed_events,
        energy_profile=EnergyProfile(),
        allow_class_time=allow_class_time,
    )

    initial = system.schedule_day()
    print_schedule(initial, "初始排程")
    print_timeline(system._at(DAY_START), system._at(DAY_END), fixed_events, initial)

    english_target = next((t.total_minutes for t in tasks if t.category == "english"), 0)
    validate_schedule_constraints(initial, fixed_events, english_target_minutes=english_target)

    if mode == "roll":
        if not current_time:
            current_time = ask("请输入 current_time (HH:MM)")
        if disruption_minutes is None:
            disruption_minutes = int(ask("请输入 disruption_minutes", "0"))

        rolled = system.rolling_reschedule(parse_time_on_date(base_day, current_time), int(disruption_minutes))
        print_schedule(rolled, "滚动重排后")
        print_timeline(system._at(DAY_START), system._at(DAY_END), fixed_events, rolled)
        validate_schedule_constraints(rolled, fixed_events, english_target_minutes=english_target)


def main() -> None:
    parser = argparse.ArgumentParser(description="精密时间管理系统 CLI")
    parser.add_argument("mode", nargs="?", choices=["plan", "roll"], help="plan 或 roll")
    parser.add_argument("--time", dest="current_time", help="roll 模式 current_time，格式 HH:MM")
    parser.add_argument("--disruption", type=int, help="roll 模式 disruption_minutes")
    parser.add_argument("--allow-class-time", choices=["y", "n"], help="是否允许上课时段插入可中断任务")
    parser.add_argument("--english", type=int, help="english_minutes")
    parser.add_argument("--math-deep", type=int, help="math_deep_minutes")
    parser.add_argument("--math-non-deep", type=int, help="math_non_deep_minutes")
    parser.add_argument("--review", type=int, help="review_minutes")
    parser.add_argument("--date", help="日期 YYYY-MM-DD")
    parser.add_argument("--config", help="JSON 配置文件路径")
    args = parser.parse_args()

    mode = args.mode or ask("请选择模式 mode (plan/roll)", "plan")

    date_str = args.date if args.date is not None else ask("请输入日期 YYYY-MM-DD（回车默认今天）", "")
    base_day = parse_date(date_str if date_str else None)

    cfg_allow = None
    cfg_tasks = None
    cfg_fixed_events = None
    if args.config:
        cfg_allow, cfg_tasks, cfg_fixed_events = load_config(Path(args.config), base_day)

    allow_raw = args.allow_class_time
    if allow_raw is None and cfg_allow is None:
        allow_raw = ask("allow_class_time? (y/n)", "y")
    allow_class_time = cfg_allow if cfg_allow is not None else (allow_raw == "y")

    if cfg_tasks is not None:
        tasks = cfg_tasks
    else:
        english = args.english if args.english is not None else int(ask("english_minutes", "120"))
        math_deep = args.math_deep if args.math_deep is not None else int(ask("math_deep_minutes", "120"))
        math_non_deep = args.math_non_deep if args.math_non_deep is not None else int(ask("math_non_deep_minutes", "180"))
        review = args.review if args.review is not None else int(ask("review_minutes", "60"))
        tasks = build_tasks(english, math_deep, math_non_deep, review)

    fixed_events = cfg_fixed_events if cfg_fixed_events is not None else build_default_fixed_events(base_day)

    run(
        mode=mode,
        base_day=base_day,
        allow_class_time=allow_class_time,
        tasks=tasks,
        fixed_events=fixed_events,
        current_time=args.current_time,
        disruption_minutes=args.disruption,
    )


if __name__ == "__main__":
    main()
