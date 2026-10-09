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
4. **命令必须吸附到档实测速度**(默认开启)。学生训练时命令恒等于所匹配 clip 的
   实测速度 —— ``command_jitter=0.3`` 乘的是 Voronoi 半宽,而本库在 vx 轴上有
   近似重复条目(``walk_0.185_*`` 与 ``walk_0.37_*`` 实测都是 0.188),同轴最近邻
   距离 ~1e-3,半宽被压到 ~4e-4,**抖动实际为 0**。于是学生只见过 25 个离散命令
   点。实测(``scripts/scan_command_space.py``,vx 轴 17 点):命令 0.118(落在
   ``stand`` 与 ``walk_0.37`` 之间)时学生只走出 0.015 m/s、步幅 0.017 m
   —— 就是"小碎步";吸附到该档实测 0.189 后变成 0.172 m/s、步幅 0.130 m。
   滑条只用来**选**哪条参考,真正执行的速度是这条参考自己的实测速度。

用法::

    uv run play Mjlab-Tracking-Flat-Pikachu-Lite-Distill \\
      --checkpoint-file <model.pt> --viewer velocity-joystick \\
      --no-terminations True

``--no-terminations True`` 是必须的:tracking 档 ``episode_length_s=10.0``,不关掉的
话回合每 10 秒重置一次,机器人会被反复瞬移回参考起点。值 ``True`` 也不能省 —— 字段
名自带 ``no_`` 前缀,tyro 视其为歧义名而要求显式取值(``play <TASK> --help`` 里写作
``--no-terminations {True,False}``)。
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
    self._shown_key: tuple[Any, ...] | None = None
    # 默认吸附 —— 学生的训练命令就是各档实测速度(见模块 docstring 第 4 点),
    # 喂格间命令等于让它外推。关掉可以复现"不吸附时档间退化"的对比。
    self._snap = True

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

      sliders: dict[str, Any] = {}
      for axis, idx, label in (
        ("vx", 0, "vx / (m/s)"),
        ("vy", 1, "vy / (m/s)"),
        ("wz", 2, "wz / (rad/s)"),
      ):
        a, b = self._slider_bounds(lo, hi, idx)
        slider = server.gui.add_slider(
          label, min=a, max=b, step=0.01, initial_value=0.0
        )
        sliders[axis] = slider

        @slider.on_update
        def _(_ev, _s=slider, _axis=axis) -> None:
          setattr(self, f"_{_axis}", float(_s.value))

      zero = server.gui.add_button("归零")

      @zero.on_click
      def _(_ev) -> None:
        # 必须写回**滑条**本身,不能只改 self._vx:滑条的位置由 viser 的 handle
        # 持有并同步给浏览器,只改 Python 属性的话命令归零了、滑条还停在原处
        # (面板上"滑条请求"与"实际执行"当场对不上)。
        # 给 handle 赋值会同步触发上面的 on_update,顺带把 self._vx 也置零,
        # 所以这里不需要再 setattr 一次。
        for s in sliders.values():
          s.value = 0.0

      snap = server.gui.add_checkbox("命令吸附到档实测速度", initial_value=self._snap)

      @snap.on_update
      def _(_ev) -> None:
        self._snap = bool(snap.value)
        self._shown_key = None  # 强制刷新状态行

      self._status_html = server.gui.add_html("")
      server.gui.add_html(
        "<div style='font-size:0.78em;line-height:1.3;opacity:0.75;"
        "padding:0 1em 0.5em;'>勾选 Enable 后生效;滑条按最近邻**选**参考,"
        "参考播完会在本 clip 内循环(不瞬移)。吸附开启时真正执行的速度是所选"
        "参考的实测速度 —— 学生只在这些离散命令上训练过。</div>"
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
    else:
      self._fold_reference()

    # 3. 吸附到该档实测速度 —— 见模块 docstring 第 4 点。必须放在匹配**之后**:
    #    匹配要用滑条的原值决定"选哪条参考",执行值再用选中那条的实测速度覆盖。
    if self._snap:
      term.velocity_command[:] = self._lib.clip_command[term.clip_ids]

    self._sync_status(target)

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
    """把"匹配到哪条参考 / 实际执行什么命令"显示到面板上。

    每步都会调,但只在 ``(clip, 滑条值)`` 变化时才写 DOM —— 否则 50 Hz 刷
    viser 的 HTML 太浪费。
    """
    if self._status_html is None:
      return
    key = (clip, round(self._vx, 3), round(self._vy, 3), round(self._wz, 3))
    if key == self._shown_key:
      return
    self._shown_key = key
    term = self._term
    assert term is not None
    name = self._lib.clip_names[clip]
    exec_cmd = self._lib.clip_command[clip].cpu()
    req = f"vx={self._vx:+.3f} vy={self._vy:+.3f} wz={self._wz:+.3f}"
    act = (
      f"vx={exec_cmd[0]:+.3f} vy={exec_cmd[1]:+.3f} wz={exec_cmd[2]:+.3f}"
      if self._snap
      else req
    )
    tag = " <span style='opacity:0.6'>(吸附)</span>" if self._snap else ""
    self._status_html.content = (
      "<div style='font-size:0.85em;line-height:1.35;padding:0 1em 0.5em;'>"
      f"匹配参考: <code>{name}</code><br/>"
      f"滑条请求: {req}<br/>"
      f"实际执行: {act}{tag}"
      "</div>"
    )
