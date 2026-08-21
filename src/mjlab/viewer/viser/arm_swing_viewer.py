"""ArmSwingViewer:手臂摆动力学消融实验的 Viser 播放器。

本任务运行在**纯腿 10 关节策略**上(手臂已从策略输入/输出剥离),手臂由 env 内的
零维副作用 ``ArmSineAction``(``envs/mdp/actions/arm_sine.py``)在每个物理子步施加
正弦摆动。本 viewer 的三根滑条直接写该 term 的幅值/频率 override:
- ``臂 pitch 摆幅 / rad`` → ``term.amp_override_pitch``
- ``臂 roll 摆幅 / rad``  → ``term.amp_override_roll``
- ``摆频 / Hz``          → ``term.freq_hz``

基类默认 ``_postprocess_actions`` 钩子(透传)保留,腿部 10 维动作原样交给策略。
"""

from mjlab.asset_zoo.robots.pikachu.pikachu_constants_base import (
  arm_pitch_amp_upper_limit,
  arm_roll_amp_upper_limit,
)
from mjlab.envs.mdp.actions.arm_sine import ArmSineAction
from mjlab.viewer.viser.viewer import ViserPlayViewer


class ArmSwingViewer(ViserPlayViewer):
  """Viser 播放器,通过滑条在线控制摆臂正弦的摆幅与频率。

  三根滑条直接改写 env 内 ``arm_sine`` term 的 ``amp_override_*`` 与 ``freq_hz``,
  不再对策略动作输出做任何改写。
  """

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._arm_pitch_amp = 0.0
    self._arm_roll_amp = 0.0
    self._arm_freq_hz = 1.0
    self._arm_sine_term: ArmSineAction | None = None

  def setup(self) -> None:
    """Build the arm-swing UI on top of the default ViserPlayViewer controls."""
    super().setup()
    self._resolve_arm_sine_term()
    self._add_arm_swing_sliders()

  def _get_arm_sine_term(self) -> ArmSineAction | None:
    """Return the env's ``arm_sine`` action term, or ``None`` if absent."""
    try:
      term = self.env.unwrapped.action_manager.get_term("arm_sine")
    except Exception:
      return None
    return term if isinstance(term, ArmSineAction) else None

  def _resolve_arm_sine_term(self) -> None:
    self._arm_sine_term = self._get_arm_sine_term()
    if self._arm_sine_term is not None:
      self._arm_freq_hz = self._arm_sine_term.freq_hz

  def _add_arm_swing_sliders(self) -> None:
    """Add the three arm-swing sliders to the Viser GUI."""
    pitch_lim = arm_pitch_amp_upper_limit()
    roll_lim = arm_roll_amp_upper_limit()

    pitch_slider = self._server.gui.add_slider(
      label="臂 pitch 摆幅 / rad",
      min=0.0,
      max=max(pitch_lim, 0.01),
      step=0.01,
      initial_value=0.0,
    )
    pitch_slider.on_update(
      lambda _ev: setattr(self, "_arm_pitch_amp", float(pitch_slider.value))
    )

    roll_slider = self._server.gui.add_slider(
      label="臂 roll 摆幅 / rad",
      min=0.0,
      max=max(roll_lim, 0.01),
      step=0.01,
      initial_value=0.0,
    )
    roll_slider.on_update(
      lambda _ev: setattr(self, "_arm_roll_amp", float(roll_slider.value))
    )

    freq_slider = self._server.gui.add_slider(
      label="摆频 / Hz",
      min=0.0,
      max=3.0,
      step=0.05,
      initial_value=self._arm_freq_hz,
    )
    freq_slider.on_update(
      lambda _ev: setattr(self, "_arm_freq_hz", float(freq_slider.value))
    )

  def _sync_arm_sine_term(self) -> None:
    """Push slider state into the env's arm_sine term before each step."""
    term = self._arm_sine_term
    if term is None:
      return
    term.set_amps(self._arm_pitch_amp, self._arm_roll_amp)
    term.freq_hz = self._arm_freq_hz

  def _execute_step(self) -> bool:
    self._sync_arm_sine_term()
    return super()._execute_step()


__all__ = ["ArmSwingViewer"]
