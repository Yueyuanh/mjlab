"""VelocityJoystickViewer:给学生蒸馏策略用的实时速度摇杆。

学生档 ``Mjlab-Tracking-Flat-Pikachu-Lite-Distill`` 的命令项是
``VelocityMatchMotionCommand`` —— 策略观测里唯一的任务指令就是 ``(vx, vy, wz)``,
命令项据此在参考库里做最近邻匹配,把该 env 的参考切到对应 clip。所以
"发速度命令 -> 机器人自动匹配运动"在播放器里天然就是三根滑条。

tracking 任务的 ``MotionCommand.create_gui``(``tracking/mdp/commands.py``)只提供
"拖帧"控件,没有速度摇杆,本文件补上这一块。

三个必须处理的点
----------------
1. **参考播完会把机器人瞬移走。** ``_update_command`` 在 ``time_steps`` 越过本
   clip 末尾时调 ``_resample_command``,后者把根位姿与关节角**写进 sim**。训练时
   这是正常的回合重置,但实时摇杆下表现为"走十来秒突然闪一下"。所以摇杆开启时,
   在 ``_postprocess_actions`` 里提前把 ``time_steps`` 折回本 clip 内,让 wrap
   永不触发。
2. **折返必须保持步态相位连续。** 学生观测里有 ``gait_phase`` =
   ``(cos(2*pi*t/P), sin(2*pi*t/P))``(``multi_motion.py``)。直接把 ``time_steps``
   写回 clip 起点会让相位突跳、策略踉跄。折返时减去**整数个周期** ``k*P``,
   ``(cos, sin)`` 完全不变。
3. **换 clip 时同样让相位对齐**,否则改命令的瞬间会踢一下。

用法::

    uv run play Mjlab-Tracking-Flat-Pikachu-Lite-Distill \\
      --checkpoint-file <model.pt> --viewer velocity-joystick --no-terminations

``--no-terminations`` 是必须的:tracking 档 ``episode_length_s=10.0``,不关掉的话
回合每 10 秒重置一次,机器人会被反复瞬移回参考起点。
"""

from __future__ import annotations

from typing import Any, cast

import torch

from mjlab.tasks.tracking.mdp.multi_motion import MultiMotionLoader
from mjlab.tasks.tracking.mdp.velocity_match import VelocityMatchMotionCommand
from mjlab.viewer.viser.viewer import ViserPlayViewer

# 滑条范围在库跨度上的外扩系数 —— 留一点余量,好让用户试到"格与格之间"。
_SLIDER_MARGIN = 1.05


class VelocityJoystickViewer(ViserPlayViewer):
  """Viser 播放器,用三根滑条给学生策略实时发速度命令。

  滑条直接写 ``VelocityMatchMotionCommand.velocity_command``,并按最近邻重新匹配
  参考 clip;参考播完时在本 clip 内做**相位连续**的折返,机器人不会瞬移。
  """

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
    self._enabled = False
    self._vx = 0.0
    self._vy = 0.0
    self._wz = 0.0
    self._term: VelocityMatchMotionCommand | None = None
    self._status_html: Any = None
    self._shown_clip: int | None = None

  # -- 初始化 -----------------------------------------------------------------

  def setup(self) -> None:
    """Build the joystick UI on top of the default ViserPlayViewer controls."""
    super().setup()
    self._resolve_term()
    self._add_joystick_gui()

  def _resolve_term(self) -> None:
    """拿到 ``motion`` 命令项;不是速度匹配型(如教师档)就保持禁用。"""
    try:
      term = self.env.unwrapped.command_manager.get_term("motion")
    except (KeyError, AttributeError):
      term = None
    if isinstance(term, VelocityMatchMotionCommand):
      self._term = term

  @property
  def _lib(self) -> MultiMotionLoader:
    """参考库。``motion`` 的静态类型是基类 ``MotionLoader``,这里窄化到多动作版。"""
    assert self._term is not None
    return cast(MultiMotionLoader, self._term.motion)

  def _add_joystick_gui(self) -> None:
    server = self._server
    with server.gui.add_folder("Command Joystick"):
      if self._term is None:
        server.gui.add_html(
          "<div style='font-size:0.85em;line-height:1.3;"
          "padding:0 1em 0.5em;'>本任务没有 <code>VelocityMatchMotionCommand</code>"
          " 命令项(教师档请用默认 viser 面板的拖帧控件)。</div>"
        )
        return

      lib = self._lib.clip_command.cpu()
      lo = lib.min(dim=0).values
      hi = lib.max(dim=0).values

      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      @enabled.on_update
      def _(_ev) -> None:
        self._enabled = bool(enabled.value)

      for axis, idx, label in (
        ("vx", 0, "vx / (m/s)"),
        ("vy", 1, "vy / (m/s)"),
        ("wz", 2, "wz / (rad/s)"),
      ):
        a, b = self._slider_bounds(lo, hi, idx)
        slider = server.gui.add_slider(
          label, min=a, max=b, step=0.01, initial_value=0.0
        )

        @slider.on_update
        def _(_ev, _s=slider, _axis=axis) -> None:
          setattr(self, f"_{_axis}", float(_s.value))

      zero = server.gui.add_button("归零")

      @zero.on_click
      def _(_ev) -> None:
        for name in ("vx", "vy", "wz"):
          setattr(self, f"_{name}", 0.0)

      self._status_html = server.gui.add_html("")
      server.gui.add_html(
        "<div style='font-size:0.78em;line-height:1.3;opacity:0.75;"
        "padding:0 1em 0.5em;'>勾选 Enable 后生效;命令按最近邻匹配参考,"
        "参考播完会在本 clip 内循环(不瞬移)。</div>"
      )
      self._sync_status(int(self._term.clip_ids[0]))

  @staticmethod
  def _slider_bounds(
    lo: torch.Tensor, hi: torch.Tensor, idx: int
  ) -> tuple[float, float]:
    """该轴的滑条范围:库跨度外扩一点;库在该轴上无变化时给个默认窗口。"""
    a, b = float(lo[idx]) * _SLIDER_MARGIN, float(hi[idx]) * _SLIDER_MARGIN
    if b - a < 1e-3:
      return -0.5, 0.5
    return a, b

  # -- 每步注入 ---------------------------------------------------------------

  def _postprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
    """在 ``policy(obs)`` 与 ``env.step()`` 之间把摇杆命令写进命令项。"""
    if self._enabled and self._term is not None:
      self._apply_command()
    return actions

  def _apply_command(self) -> None:
    term = self._term
    assert term is not None
    cmd = torch.tensor(
      [[self._vx, self._vy, self._wz]], device=term.device, dtype=torch.float32
    )

    # 1. 命令写进去 —— 策略下一个 obs 就能看到。
    term.velocity_command[:] = cmd

    # 2. 命令 -> 最近邻 clip。落在同一条 clip 的格子里就不换,避免来回抖。
    target = int(term.match_clip_id(cmd)[0])
    if bool((term.clip_ids != target).any()):
      term.clip_ids[:] = target
      term.time_steps[:] = self._clip_entry(target)
      self._sync_status(target)
    else:
      self._fold_reference()

  def _clip_entry(self, clip: int) -> int:
    """进入某条 clip 的入口帧(与 ``_sample_time_steps`` 的 start 分支一致)。"""
    term = self._term
    assert term is not None
    # ``_start_frame`` 是 ``cfg.start_frame`` 按参考长度 clamp 后的结果,没有公开
    # 访问器;这里只读,不改。
    return int(self._lib.clip_starts[clip]) + term._start_frame

  def _fold_reference(self) -> None:
    """把 ``time_steps`` 折回本 clip 内,让 wrap 永不触发。

    折返量取**整数个周期** ``k*P``:``gait_phase`` 是 ``2*pi*t/P`` 的 (cos, sin),
    减去整数个周期后相位完全不变,策略看不出接缝。
    """
    term = self._term
    assert term is not None

    starts = self._lib.clip_starts[term.clip_ids]
    ends = self._lib.clip_ends[term.clip_ids]
    periods = self._lib.clip_periods[term.clip_ids].clamp(min=1.0)
    entry = (starts + term._start_frame).float()

    # ``_update_command`` 下一步会 +1,所以按 t+1 判断是否将要越界。
    over = (term.time_steps + 1) >= ends
    if not bool(over.any()):
      return

    t = term.time_steps[over].float()
    k = ((t - entry[over]) / periods[over]).floor()
    folded = t - k * periods[over]
    # 兜底:周期估计离谱时退回 clip 入口,保证索引始终落在本 clip 内。
    lo = entry[over]
    hi = (ends[over] - 2).float()
    term.time_steps[over] = torch.maximum(torch.minimum(folded, hi), lo).long()

  # -- 状态显示 ---------------------------------------------------------------

  def _sync_status(self, clip: int) -> None:
    """把"命令匹配到了哪条参考"显示到面板上。"""
    if self._status_html is None or clip == self._shown_clip:
      return
    self._shown_clip = clip
    term = self._term
    assert term is not None
    name = self._lib.clip_names[clip]
    center = self._lib.clip_command[clip].cpu()
    self._status_html.content = (
      "<div style='font-size:0.85em;line-height:1.35;padding:0 1em 0.5em;'>"
      f"匹配参考: <code>{name}</code><br/>"
      f"该档实测: vx={center[0]:+.3f} vy={center[1]:+.3f} wz={center[2]:+.3f}"
      "</div>"
    )
