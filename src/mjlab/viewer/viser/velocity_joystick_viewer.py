"""VelocityJoystickViewer:给学生蒸馏策略用的实时速度摇杆。

学生档 ``Mjlab-Tracking-Flat-Pikachu-Lite-Distill`` 的命令项是
``VelocityMatchMotionCommand`` —— 策略观测里唯一的任务指令就是 ``(vx, vy, wz)``,
命令项据此在参考库里做最近邻匹配,把该 env 的参考切到对应 clip。所以
"发速度命令 -> 机器人自动匹配运动"在播放器里天然就是三根滑条。

tracking 任务的 ``MotionCommand.create_gui``(``tracking/mdp/commands.py``)只提供
"拖帧"控件,没有速度摇杆,本文件补上这一块。

面板上有两套控件:**二维摇杆**(中间一个圆点,可全向拖)与**三根滑条**。两者写
的是同一个命令 —— 摇杆拖动时是**对着滑条**派发合成鼠标事件(见 ``_add_pad_gui``),
所以命令与滑条永远一致,滑条也能反过来当微调。摇杆的外圈 = 该轴的最大命令(库内
最快的那条参考),小圈 = 死区(圈内命令为 0),打到圈外一律裁到最大。

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

import json
from typing import Any, cast

import torch
from viser import _messages

from mjlab.tasks.tracking.mdp.multi_motion import MultiMotionLoader
from mjlab.tasks.tracking.mdp.velocity_match import VelocityMatchMotionCommand
from mjlab.viewer.viser.viewer import ViserPlayViewer

# 滑条范围在库跨度上的外扩系数 —— 留一点余量,好让用户试到"格与格之间"。
_SLIDER_MARGIN = 1.05

# 滑条步长。摇杆也走滑条,所以这也是摇杆的分辨率 —— 对本库够用,学生本来就只在
# 25 条离散参考上训练过(见模块 docstring 第 4 点)。
_SLIDER_STEP = 0.01

# 摇杆容器在页面里的 id:``add_html`` 建出这个空壳,注入的脚本往里画 SVG。
_PAD_ID = "pikachu-joy-pad"

# 死区(面板上那个小圈)半径,占摇杆最大行程的比例。圈内命令为 0,圈外按
# ``(mag - dead) / (1 - dead)`` 重标到 0..1,这样出圈时命令是连续的。
_PAD_DEADZONE = 0.16

# 转向键。左转 = wz 正(npz 约定)。转向与平移互不影响,所以单独一组按钮。
_TURN_LEFT = "⟲ 左转"
_TURN_STOP = "■ 停"
_TURN_RIGHT = "⟳ 右转"
_TURN_OPTIONS = (_TURN_LEFT, _TURN_STOP, _TURN_RIGHT)

# 判定"这条参考基本不转 / 基本不平移"的阈值 —— 选转向档时用,见 _turn_table。
_PAD_STILL_WZ = 0.05  # rad/s
_PAD_STILL_V = 0.05  # m/s

# 摇杆空壳。注入脚本会用 SVG 覆盖里面的内容;脚本没跑起来时这行提示就是兜底。
_PAD_HTML = (
  f"<div id='{_PAD_ID}' style='padding:0 1em 0.4em;'>"
  "<div style='font-size:0.78em;line-height:1.35;opacity:0.7;'>"
  "摇杆加载中…(若一直是这一行,说明浏览器没有执行注入脚本;下面的滑条仍可用)"
  "</div></div>"
)

_PAD_HINT_HTML = (
  "<div style='font-size:0.78em;line-height:1.35;opacity:0.75;"
  "padding:0 1em 0.5em;'>"
  "拖圆点发命令:上=前进,下=后退,左=向左平移,右=向右平移。<br/>"
  "打到外圈 = 该轴最大命令(库内最快的那条参考),小圈内是死区(命令 0)。<br/>"
  "双击摇杆回中;松手后命令保持。转向键只改 wz,与平移互不影响。"
  "</div>"
)

# 摇杆脚本。经 ``RunJavascriptMessage`` 注入到页面主线程 —— 这正是 viser 自己发
# plotly.min.js 用的那条通道(``viser/_gui_api.py``)。脚本自己会重试挂载(见
# ``mount``/``WATCHDOG_MS``):客户端连上来时面板 DOM 可能还没渲染完。
_PAD_JS = """
(function () {
  "use strict";
  var CFG = __CFG__;
  // 盘面几何,单位就是 SVG 的 viewBox 坐标(0..S)。
  var S = 200, C = 100, R = 78, KNOB_R = 16;
  var DR = R * CFG.dead;
  var WATCHDOG_MS = 400;
  // 每根轴最快多久发一次。见 tick() 的注释 —— 必须大于 viser 的 50ms 节流窗口。
  var SEND_MS = 55;

  var state = {
    host: null, svg: null, knob: null, readout: null,
    dragging: false, nx: 0, ny: 0,
    target: {}, sent: {}, order: 0
  };

  function norm(s) {
    return String(s == null ? "" : s).replace(/\\s+/g, " ").trim();
  }

  // 用 <label> 的 for 反查滑条根节点:viser 的 handle 不暴露 uuid,但 DOM 里
  // LabeledInput 就是这么把标签与控件关联起来的。
  function findSlider(label) {
    var labels = document.querySelectorAll("label");
    for (var i = 0; i < labels.length; i++) {
      if (norm(labels[i].textContent).indexOf(label) === 0) {
        var id = labels[i].getAttribute("for");
        if (id) {
          var el = document.getElementById(id);
          if (el) return el;
        }
      }
    }
    return null;
  }

  var rootCache = {};
  function sliderRoot(axis) {
    var hit = rootCache[axis.key];
    if (!hit || !document.contains(hit)) {
      hit = findSlider(axis.label);
      rootCache[axis.key] = hit;
    }
    return hit;
  }

  // 读滑条当前值。Mantine 的滑块把值写在 role=slider 的 aria-valuenow 上。
  function readAxis(axis) {
    var root = sliderRoot(axis);
    if (!root) return null;
    var th = root.querySelector('[role="slider"]');
    if (!th) return null;
    var v = parseFloat(th.getAttribute("aria-valuenow"));
    return isNaN(v) ? null : v;
  }

  function quant(axis, value) {
    var step = axis.step > 0 ? axis.step : 1e-6;
    var q = Math.round(value / step) * step;
    q = Math.min(Math.max(q, axis.min), axis.max);
    return Math.round(q * 1e6) / 1e6;
  }

  // 把 q 写进**真实的滑条**:对着 trackContainer 派发合成 mousedown,再在
  // document 上补一个 mouseup 收尾。Mantine 的 use-move 只认 MouseEvent
  // (PointerEvent 派下去值不动),而且没有 input[type=range] 可以走原生 setter。
  // 绕这一圈是为了复用 viser 支持的更新路径:滑条 onChange -> GuiUpdateMessage
  // -> Python on_update,于是命令与滑条显示天然同步。
  function emit(axis, q) {
    var root = sliderRoot(axis);
    if (!root) return;
    var tc = root.querySelector(".mantine-Slider-trackContainer") || root;
    var r = tc.getBoundingClientRect();
    if (!(r.width > 0)) return;
    var f = (q - axis.min) / (axis.max - axis.min);
    var x = r.left + f * r.width;
    var y = r.top + r.height / 2;
    tc.dispatchEvent(new MouseEvent("mousedown",
      { bubbles: true, cancelable: true, view: window,
        clientX: x, clientY: y, button: 0, buttons: 1 }));
    document.dispatchEvent(new MouseEvent("mouseup",
      { bubbles: true, cancelable: true, view: window,
        clientX: x, clientY: y, button: 0, buttons: 0 }));
    state.sent[axis.key] = q;
  }

  // 一次只发**一根**轴,两根轮流。
  //
  // 不能一次把两根都发出去:viser 客户端的 GUI 更新走的是
  // ``WebsocketUtils.makeThrottledMessageSender``,50ms 节流,而且**整个面板共用
  // 一个 ``latestMessage`` 槽位**(``ControlPanel/Generated.tsx`` 只建一个 sender
  // 塞进 context)。窗口内只有第一条会立刻出去、最后一条会在窗口末尾补发,中间的
  // 全被覆盖 —— 连发两根时必有一根丢。实测:摇杆拖到最左,滑条显示 vx=0,服务端
  // 的 handle 却还停在拖动第一帧的 0.13。
  //
  // 所以这里自己排队:55ms > 50ms,保证每次 emit 都落在一个**空**窗口的第一条上,
  // 立刻发出去、不参与节流。代价是每根轴约 110ms 更新一次 —— 对速度命令足够。
  function tick() {
    var axes = CFG.axes;
    for (var n = 0; n < axes.length; n++) {
      var i = (state.order + n) % axes.length;
      var ax = axes[i];
      var q = state.target[ax.key];
      if (q === undefined || q === state.sent[ax.key]) continue;
      emit(ax, q);
      state.order = (i + 1) % axes.length;
      return;
    }
  }

  // 盘面坐标 -> 命令。屏幕右 = -vy(y 轴正向是机器人**左侧**),屏幕上 = +vx。
  function command() {
    var hyp = Math.sqrt(state.nx * state.nx + state.ny * state.ny);
    var t = hyp <= CFG.dead ? 0 : (hyp - CFG.dead) / (1 - CFG.dead);
    if (t > 1) t = 1;
    return [-state.ny * t * CFG.max[0], -state.nx * t * CFG.max[1]];
  }

  function fmt(x) {
    return (x >= 0 ? "+" : "-") + Math.abs(x).toFixed(3);
  }

  function paint() {
    if (state.knob) {
      state.knob.setAttribute("cx", (C + state.nx * R).toFixed(1));
      state.knob.setAttribute("cy", (C + state.ny * R).toFixed(1));
    }
    if (state.readout) {
      var v = command();
      state.readout.textContent = "vx=" + fmt(v[0]) + " vy=" + fmt(v[1]);
    }
  }

  // 只登记目标值,真正发送交给 tick() —— 在 pointermove 里直接发的话,两次移动
  // 会挤进同一个节流窗口,又回到互相覆盖的老问题。
  function commit() {
    var v = command();
    for (var i = 0; i < CFG.axes.length; i++) {
      state.target[CFG.axes[i].key] = quant(CFG.axes[i], v[i]);
    }
  }

  function drag(ev) {
    var r = state.svg.getBoundingClientRect();
    if (!(r.width > 0)) return;
    var nx = ((ev.clientX - r.left) / r.width * S - C) / R;
    var ny = ((ev.clientY - r.top) / r.height * S - C) / R;
    // 裁到单位圆:拖出圈外时旋钮贴边,命令停在该轴最大值。
    var hyp = Math.sqrt(nx * nx + ny * ny);
    if (hyp > 1) { nx /= hyp; ny /= hyp; }
    state.nx = nx;
    state.ny = ny;
    commit();
    paint();
  }

  function wire() {
    var svg = state.svg;
    if (!svg) return;
    svg.addEventListener("pointerdown", function (ev) {
      if (ev.button !== 0) return;
      ev.preventDefault();
      state.dragging = true;
      state.sent = {};
      try { svg.setPointerCapture(ev.pointerId); } catch (e) {}
      svg.style.cursor = "grabbing";
      drag(ev);
    });
    svg.addEventListener("pointermove", function (ev) {
      if (!state.dragging) return;
      ev.preventDefault();
      drag(ev);
    });
    var stop = function (ev) {
      if (!state.dragging) return;
      state.dragging = false;
      svg.style.cursor = "grab";
      try { svg.releasePointerCapture(ev.pointerId); } catch (e) {}
    };
    svg.addEventListener("pointerup", stop);
    svg.addEventListener("pointercancel", stop);
    svg.addEventListener("dblclick", function (ev) {
      ev.preventDefault();
      state.nx = 0;
      state.ny = 0;
      state.sent = {};
      commit();
      paint();
    });
  }

  function build(host) {
    host.innerHTML =
      '<div style="display:flex;flex-direction:column;align-items:center;' +
             'gap:2px;">' +
        '<svg viewBox="0 0 200 200" width="176" height="176" ' +
             'style="touch-action:none;cursor:grab;display:block;' +
             'user-select:none;">' +
          '<circle cx="100" cy="100" r="78" fill="none" stroke="currentColor" ' +
                  'stroke-width="1.5" opacity="0.35"/>' +
          '<circle cx="100" cy="100" r="' + DR.toFixed(1) + '" fill="none" ' +
                  'stroke="currentColor" stroke-width="1.2" opacity="0.4" ' +
                  'stroke-dasharray="3 3"/>' +
          '<line x1="26" y1="100" x2="174" y2="100" stroke="currentColor" ' +
                'stroke-width="1" opacity="0.18"/>' +
          '<line x1="100" y1="26" x2="100" y2="174" stroke="currentColor" ' +
                'stroke-width="1" opacity="0.18"/>' +
          '<text x="100" y="18" fill="currentColor" opacity="0.55" ' +
                'font-size="12" text-anchor="middle">\\u524d</text>' +
          '<text x="100" y="194" fill="currentColor" opacity="0.55" ' +
                'font-size="12" text-anchor="middle">\\u540e</text>' +
          '<text x="10" y="105" fill="currentColor" opacity="0.55" ' +
                'font-size="12" text-anchor="middle">\\u5de6</text>' +
          '<text x="190" y="105" fill="currentColor" opacity="0.55" ' +
                'font-size="12" text-anchor="middle">\\u53f3</text>' +
          '<circle class="pk-knob" cx="100" cy="100" r="16" ' +
                  'fill="currentColor" fill-opacity="0.18" ' +
                  'stroke="currentColor" stroke-width="2"/>' +
        '</svg>' +
        '<div class="pk-read" style="font-size:0.8em;opacity:0.8;' +
             'font-variant-numeric:tabular-nums;">vx=+0.000 vy=+0.000</div>' +
      '</div>';
    state.host = host;
    state.svg = host.querySelector("svg");
    state.knob = host.querySelector(".pk-knob");
    state.readout = host.querySelector(".pk-read");
    host.__pkSvg = state.svg;
    wire();
    paint();
  }

  // 面板 DOM 由 React 管,注入的内容可能被重渲染冲掉;每次挂载失败或节点被换掉
  // 就重建一次。客户端刚连上时 DOM 还没渲染,也靠它兜住。
  function mount() {
    var host = document.getElementById(CFG.padId);
    if (!host) return;
    var svg = host.querySelector("svg");
    if (svg && host.__pkSvg === svg) return;
    build(host);
  }

  // 反过来把滑条的值同步到旋钮上 —— 「归零」按钮与转向键都是 Python 侧直接写
  // 滑条的,旋钮得跟着回中,否则盘面显示与实际命令对不上。
  //
  // 同时**认账**:别人写进去的值就等于服务端已经有了,记进 target/sent,免得
  // 调度器以为还欠一次发送、把刚归零的命令又推回去。但只在没有待发值时才认
  // (``target === sent``)—— 否则会把摇杆刚拖出来、还没发出去的值抹掉。
  function sync() {
    if (state.dragging || !state.knob) return;
    var vx = readAxis(CFG.axes[0]);
    var vy = readAxis(CFG.axes[1]);
    if (vx === null || vy === null) return;
    for (var i = 0; i < CFG.axes.length; i++) {
      var ax = CFG.axes[i];
      var v = i === 0 ? vx : vy;
      if (state.target[ax.key] === state.sent[ax.key]) {
        state.target[ax.key] = v;
        state.sent[ax.key] = v;
      }
    }
    var a = CFG.max[0] ? -vx / CFG.max[0] : 0;  // ny * t
    var b = CFG.max[1] ? -vy / CFG.max[1] : 0;  // nx * t
    var t = Math.sqrt(a * a + b * b);
    if (t < 1e-4) {
      state.nx = 0;
      state.ny = 0;
    } else {
      var hyp = CFG.dead + Math.min(t, 1) * (1 - CFG.dead);
      state.nx = b / t * hyp;
      state.ny = a / t * hyp;
    }
    paint();
  }

  if (window.__pikachuJoyPad) {
    window.__pikachuJoyPad.mount();
    window.__pikachuJoyPad.sync();
    return;
  }
  window.__pikachuJoyPad = { mount: mount, sync: sync, state: state };
  mount();
  sync();
  setInterval(tick, SEND_MS);
  setInterval(function () {
    mount();
    sync();
  }, WATCHDOG_MS);
})();
"""


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
    # 摇杆按钮要经由**滑条 handle**写命令(理由见 _set_command),所以两者都
    # 挂到实例上:滑条 handle 本身,以及每根滑条的范围(夹取用)。
    self._sliders: dict[str, Any] = {}
    self._slider_range: dict[str, tuple[float, float]] = {}
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

      # 摇杆排在滑条**前面**:创建顺序就是面板里的显示顺序,摇杆是主控件。
      self._add_pad_gui(server)

      sliders: dict[str, Any] = {}
      for axis, idx, label in (
        ("vx", 0, "vx / (m/s)"),
        ("vy", 1, "vy / (m/s)"),
        ("wz", 2, "wz / (rad/s)"),
      ):
        a, b = self._slider_bounds(lo, hi, idx)
        slider = server.gui.add_slider(
          label, min=a, max=b, step=_SLIDER_STEP, initial_value=0.0
        )
        sliders[axis] = slider
        self._slider_range[axis] = (a, b)

        @slider.on_update
        def _(_ev, _s=slider, _axis=axis) -> None:
          setattr(self, f"_{_axis}", float(_s.value))

      self._sliders = sliders
      # 注入摇杆脚本必须在**滑条建好之后** —— 它要把每根滑条的范围写进 CFG。
      self._install_pad_js()

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
        "padding:0 1em 0.5em;'>勾选 Enable 后生效;摇杆与滑条都按最近邻**选**参考,"
        "参考播完会在本 clip 内循环(不瞬移)。吸附开启时真正执行的速度是所选"
        "参考的实测速度 —— 学生只在这些离散命令上训练过。</div>"
      )
      self._sync_status(int(self._term.clip_ids[0]))

  # -- 摇杆 -------------------------------------------------------------------

  def _add_pad_gui(self, server: Any) -> None:
    """二维摇杆面板:一个空壳 + 一组转向键,真正画摇杆的是注入的脚本。

    viser 1.0.27 既没有二维摇杆控件,也没有自定义组件接口 —— 客户端组件是打包好
    的 React,加新控件只能改 viser 源码。这里走 viser **自己**用的那条注入通道:
    ``RunJavascriptMessage``(``viser/_gui_api.py`` 发 plotly.min.js 就是它),
    在页面主线程里建一个 SVG 摇杆,拖动时对着**真实的滑条**派发合成 ``mousedown``。

    为什么绕道滑条、而不是另开一条回传通道:websocket 跑在 Web Worker 里、走
    msgpack,Python 侧拿不到页面变量;viser 的 HTTP 服务又只发静态文件,没有自定义
    路由,``fetch()`` 也回不来。合成鼠标事件让摇杆**复用 viser 支持的更新路径**
    (滑条 onChange -> GuiUpdateMessage -> Python on_update),代价只是分辨率被滑条
    的 ``step`` 吃掉,而这对本库无所谓。

    转向单独一组按钮、不占摇杆:摇杆只发 (vx, vy),让"走"与"转"互不干扰。
    """
    with server.gui.add_folder("摇杆"):
      server.gui.add_html(_PAD_HTML)
      turn_group = server.gui.add_button_group("转向", list(_TURN_OPTIONS))

      def _on_turn_click(_ev) -> None:
        self._on_turn(str(turn_group.value), self._turn_table())

      turn_group.on_click(_on_turn_click)
      server.gui.add_html(_PAD_HINT_HTML)

  def _install_pad_js(self) -> None:
    """把摇杆脚本发给浏览器。

    连上来的客户端才需要脚本,所以走 ``on_client_connect`` —— 它也会对**已经连上**
    的客户端补跑一次(见 ``ViserServer.on_client_connect``),面板刚建好时就生效。
    那一刻面板 DOM 可能还没渲染完,脚本自己的重试挂载兜住这个窗口。
    """
    cfg = {
      "padId": _PAD_ID,
      "dead": _PAD_DEADZONE,
      "axes": [
        {
          "key": axis,
          "label": axis,
          "min": self._slider_range[axis][0],
          "max": self._slider_range[axis][1],
          "step": _SLIDER_STEP,
        }
        for axis in ("vx", "vy")
      ],
      "max": [self._pad_axis_max("vx"), self._pad_axis_max("vy")],
    }
    message = _messages.RunJavascriptMessage(
      source=_PAD_JS.replace("__CFG__", json.dumps(cfg))
    )

    def _send(client: Any) -> None:
      # ``_websock_connection`` 是 viser 的私有属性,但没有公开的"给单个客户端发
      # 消息"入口;取不到就放弃 —— 摇杆没装上时滑条仍然能用。
      conn = getattr(client, "_websock_connection", None)
      if conn is not None:
        conn.queue_message(message)

    self._server.on_client_connect(_send)

  def _pad_axis_max(self, axis: str) -> float:
    """摇杆打到外圈时该轴发多大命令 —— 库内该轴的最大绝对值。

    再往外没有参考可选(最近邻还是那一条),所以"裁切到最大值"就裁在这里。取
    绝对值是为了让盘面左右对称;``wz`` 不在盘面上,库内本来也不对称。
    """
    idx = {"vx": 0, "vy": 1}[axis]
    return float(self._lib.clip_command[:, idx].abs().max())

  def _turn_table(self) -> tuple[float, float]:
    """(左转, 右转) 用多大角速度:取**几乎不平移**的参考里 |wz| 最大的两条。"""
    cmd = self._lib.clip_command.cpu()
    wz = cmd[:, 2]
    still = cmd[:, :2].norm(dim=1) < _PAD_STILL_V
    wz = torch.where(still, wz, torch.zeros_like(wz))
    return float(wz.max()), float(wz.min())

  def _on_turn(self, key: str, table: tuple[float, float]) -> None:
    wz = {_TURN_LEFT: table[0], _TURN_STOP: 0.0, _TURN_RIGHT: table[1]}.get(key)
    if wz is None:
      return
    self._set_command(wz=wz)

  def _set_command(self, **axes: float) -> None:
    """把命令写进面板 —— **必须经由滑条 handle**。

    滑条的位置由 viser 的 handle 持有并同步给浏览器,只改 ``self._vx`` 会让命令
    与滑条当场脱节(「归零」按钮踩过这个坑)。给 handle 赋值会**同步**触发它的
    ``on_update``,顺带把 ``self._vx`` 也更新掉,所以这里不用再 setattr 一次。
    """
    for axis, value in axes.items():
      slider = self._sliders.get(axis)
      if slider is None:
        # 只有面板没建起来(比如命令项不是速度匹配型)时才会走到这里。
        setattr(self, f"_{axis}", float(value))
        continue
      # 目标值本来就取自库跨度内,而滑条范围又外扩了 5%,正常不会越界。夹一下是
      # 防将来换库:越界时浏览器端会夹住**显示值**,于是滑条与命令又对不上。
      lo, hi = self._slider_range.get(axis, (float("-inf"), float("inf")))
      slider.value = min(max(float(value), lo), hi)

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
