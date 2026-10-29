"""Static contract tests for the standalone Metrics CSV dashboard."""

from pathlib import Path

DASHBOARD = Path("scripts/plot_logs/metrics_csv_dashboard.html")


def test_video_click_and_loop_share_the_data_clock() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  assert "syncVideoElement.addEventListener('click'" in content
  assert 'id="video-loop"' in content and 'id="plot-loop"' in content
  assert "function restartPlaybackLoop(driver)" in content
  assert "restartPlaybackLoop('plot')" in content
  assert "restartPlaybackLoop('foot')" in content
  assert "restartPlaybackLoop(SYNC_MEDIA.driver)" in content
  assert "generation!==SYNC_MEDIA.playGeneration" in content


def test_sole_3d_uses_logged_normal_components_not_euler_angles() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  assert "function soleNormalAt(side,index)" in content
  assert "Math.sin(pitch*rad),-Math.sin(roll*rad),Math.cos(tilt*rad)" in content
  assert "Math.abs(Math.hypot(...n)-1)>.002" in content
  assert (
    "registerFootFrame(chart,index=>{frame=index;chart.__sceneRedraw();});" in content
  )
  assert "不含实测 yaw / 位置" in content
  assert "['foot-posture-3d-chart',renderFootPosture3d]" in content


def test_friction_statistics_exclude_invalid_frames_and_keep_unknown_direction() -> (
  None
):
  content = DASHBOARD.read_text(encoding="utf-8")
  assert "gaitContactState(contacts[i])!==1" in content
  assert "fn<=1e-3" in content and "ft<0" in content
  assert "ft/(m*fn)" in content
  assert "rows.filter(row=>Number.isFinite(row.util))" in content
  assert "整环表示方向未知" in content
  assert "有效帧占比，不等于已打滑或时长占比" in content
  assert "['foot-friction-load-chart',renderFootFrictionLoad]" in content
  assert "['foot-friction-cone-chart',renderFootFrictionCone]" in content


def test_compact_import_and_single_bottom_playback_dock() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  assert 'id="import-csv"' in content and ">导入 CSV</button>" in content
  assert 'id="drop-zone"' not in content
  assert 'class="playback-dock"' in content
  assert "position:fixed; inset:auto 0 0" in content
  assert content.count('type="range"') == 1
  assert "data-pressure-time" not in content
  assert "player.widgets.push({play:byId('plot-play')" in content
  assert "function applyRangeValues()" in content
  assert 'value="1.5">1.5×' in content and 'value="4">4×' in content
  assert "body.has-playback" in content


def test_shared_video_playback_and_live_transport_contract() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  for element in ("sync-video", "video-stage", "video-unpin", "m-path"):
    assert f'id="{element}"' in content
  for function in ("toggleFloatingVideo", "togglePlotPlayback", "refreshLiveSources"):
    assert f"function {function}" in content
  assert "/api/live-metrics?" in content
  assert "csv_sample" in content and "video_time_s" in content
  assert "PLOT_FOOT_BRIDGE" in content
  assert "white-space:normal" in content


def test_metrics_csv_dashboard_contains_core_analysis_features() -> None:
  """The self-contained page should preserve its CSV/debug/export contract."""
  content = DASHBOARD.read_text(encoding="utf-8")

  assert 'id="file-input"' in content
  assert "function parseCsv" in content
  assert "joint_position_rad/" in content
  assert "joint_velocity_rad_s/" in content
  assert "joint_torque_nm/" in content
  assert (
    "const CANONICAL_JOINT_ORDER = ['hip_pitch','hip_roll','hip_yaw','knee'" in content
  )
  assert "function saveOverviewPng" in content
  assert "function savePairsPng" in content
  assert "function saveSelectedCsv" in content
  assert "1920,3780" in content
  assert "RMS" in content
  assert "P95" in content


def test_foot_categories_skip_mixed_unit_overview() -> None:
  """Foot telemetry should use its shape- and unit-specific charts only."""
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "const isFootCategory=FOOT_CATEGORIES.has(ACTIVE);" in content
  assert "overviewExport.disabled=isFootCategory||!DATA;" in content
  assert "if(FOOT_CATEGORIES.has(ACTIVE)){" in content
  assert "不导出混合量纲总览" in content


def test_dashboard_hides_redundant_panels_for_foot_views() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert 'id="open-compare"' in content
  assert "保存全部为 PDF" in content
  assert (
    "document.getElementById('stats-panel').hidden=isFootCategory||isVelocity;"
    in content
  )
  assert "if(isFootCategory||isVelocity){\n    if(OVERVIEW)" in content
  assert "document.getElementById('ref-gallery').hidden=!isTn||special;" in content


def test_dashboard_layout_keeps_foot_chart_heights_and_compact_summary() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert '#foot-panel :where(div[id$="-chart"])' in content
  assert "#foot-geom-heat-time-chart { height:440px; }" in content
  assert '<details class="foot-summary">' in content
  assert (
    ".foot-geometry-pair { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }"
    in content
  )
  assert 'id="foot-pressure-fields"' in content


def test_dashboard_resizes_all_specialized_charts() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "for(const chart of Object.values(FOOT_CHARTS))chart.resize();" in content
  assert "POWER_PIE_TOTAL?.resize();" in content
  assert '.nav-group[data-group="joint"]' in content
  assert "key===FOOT_TOP_KEY?FOOT_CATEGORIES.has(ACTIVE):key===ACTIVE" in content


def test_foot_charts_skip_missing_columns() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function footViewAvailable(id)" in content
  assert "const available=views.filter(([id])=>footViewAvailable(id));" in content
  assert "foot-empty" in content
  assert "foot-pressure-fields').hidden=" in content


def test_slip_and_pressure_views_are_visually_grouped() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert 'id="foot-slip-contact-group"' in content
  assert 'id="foot-machine-group"' in content
  assert 'id="foot-joint-impact-group"' in content
  assert 'id="foot-slip-chart"' in content
  assert "let FOOT_PRESSURE_VIEW='space';" in content
  assert "FOOT_PRESSURE_VIEW===view" in content
  assert "function setFootPressureView(view)" in content
  assert 'data-pressure-view="time"' in content
  assert ".foot-slip-pair { grid-template-columns:minmax(0,1fr); }" in content


def test_foot_parameters_have_measurement_and_reading_help() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "const FOOT_METRIC_HELP={" in content
  assert "function footMetricHelp(name)" in content
  assert "const helpText=footMetricHelp(name);" in content
  assert "help.textContent='?'" in content
  assert "help.title=helpText" in content
  assert "测量：" in content
  assert "表征：" in content
  assert "读图：" in content


def test_foot_charts_can_be_zoomed_and_restored() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function attachFootChartZoom(el)" in content
  assert "function openFootChartZoom(el,group)" in content
  assert "function closeFootChartZoom()" in content
  assert "function enableFootChartNavigation(chart,el)" in content
  assert "zoomOnMouseWheel:true,moveOnMouseMove:true" in content
  assert "document.addEventListener('pointermove'" in content
  assert "foot-chart-zoom-open" in content
  assert "echarts.getInstanceByDom(el)?.resize()" in content
  assert '<div id="foot-chart-zoom" class="foot-chart-zoom" hidden>' in content


def test_foot_status_is_in_motion_nav_group() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert (
    "{id:'motion',label:'运动与任务',keys:['velocity','base','command','reward','contact',FOOT_TOP_KEY,'other']}"
    in content
  )


def test_sole_pressure_keeps_physical_aspect_when_resized() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function footPressureGrids(el,top,bottom)" in content
  assert "const width=height*56/80;" in content
  assert "keepFootPressureAspect(chart,el,64,68);" in content
  assert "keepFootPressureAspect(chart,el,52,60);" in content


def test_rated_torque_is_separate_from_simulation_limits() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "const MOTOR_RATED_TORQUE = {J3507:0.8,J4310:3.5};" in content
  assert "canonicalJoint(clean)==='ankle'?'J3507':'J4310'" in content
  assert "function ratedTorqueSeries" in content
  assert "lineStyle:{type:'dashed'" in content
  assert "motorTypeFor(joint)==='J4310'?'dm4310':'dm3507'" in content
  assert "positiveOnly:TN_VIEW==='magnitude'" in content


def test_explicit_xml_motor_mapping_can_override_defaults() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert 'id="motor-xml-input"' in content
  assert "function parseMotorXml(text)" in content
  assert "actuator > *[joint]" in content
  assert "custom > text[name]" in content
  assert "XML 未提供明确电机型号；保持现有映射" in content
  assert "function resetMotorMapping()" in content


def test_joint_totals_and_empty_pairs_are_not_rendered() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function unnecessaryJointTotal(name)" in content
  assert "if(unnecessaryJointTotal(name))continue;" in content
  assert "meta.category==='joint_power'&&!meta.side" in content
  assert "function visiblePairJoints(key=ACTIVE)" in content
  assert "hasMetricSamples(name,start,end)" in content
  assert "section.hidden=!joints.length;" in content
  assert "const joints=visiblePairJoints(key),[start,end]=rangeIndices();" in content
  assert "gap=joints.length?92/joints.length:0" in content


def test_tn_choices_require_real_samples_not_reference_curves() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "if(!hasTnSamples(side,joint))continue;" in content
  assert "&&hasTnSamples(side,joint,start,end,true)" in content
  assert "if(!TN_CHARTS.length)empty(" in content
  assert "if(powerCols.length)DATA.values[`${PREFIX}total`] = POWER.total;" in content


def test_pressure_panels_swap_positions_without_relabeling_data() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "return [1,0].map(i=>({left:" in content
  assert content.count("text:'左脚 · +x ↑',left:'67%'") == 2
  assert content.count("text:'右脚 · +x ↑',left:'20%'") == 2
  assert "const side=point.seriesIndex===1?'right':'left';" in content


def test_tn_exports_use_the_same_chart_options_as_the_screen() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function tnChartOptions()" in content
  assert "for(const {joint,option} of tnChartOptions())" in content
  assert "return tnChartOptions().map" in content
  assert content.count("if(ACTIVE==='joint_tn')return saveTnPng();") == 2
  assert "document.getElementById('save-pairs').disabled=!TN_CHARTS.length;" in content
  assert "'_joint_tn_left_right.png'" in content


def test_pdf_handles_virtual_tn_fields_before_scalar_statistics() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  pdf_loop = content[content.index("async function buildAllPdf()") :]

  assert pdf_loop.index("if(key==='joint_tn')") < pdf_loop.index(
    "overviewOption(names,true,key)"
  )
  assert "const entries=tnExportOptions();" in pdf_loop
  assert "T-N 左右对比" in pdf_loop
  assert "for(const img of root.querySelectorAll('img'))" in pdf_loop
  assert "await img.decode();" in pdf_loop
  assert "window.print();" in pdf_loop


def test_export_failures_are_visible_and_release_busy_state() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "T-N PNG 导出失败：" in content
  assert "PDF 导出失败：" in content
  assert (
    "finally{PDF_EXPORTING=false;button.disabled=!DATA;renderMotorConfig();syncGaitOverlayUi();}"
    in content
  )
  assert "catch(err){cleanup();reject(err);}" in content


def test_velocity_charts_group_tracking_by_direction_and_errors_by_unit() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert 'id="velocity-panel"' in content
  assert "function velocityGroups()" in content
  assert "add('tracking_'+axis" in content
  assert "add('error_linear','线速度误差','m/s'" in content
  assert "add('error_angular','角速度误差','rad/s'" in content
  assert "add('position','世界坐标位置','m'" in content
  assert "add('standing','站立标志','0 / 1'" in content
  assert "for(const chart of VELOCITY_CHARTS)chart.resize();" in content
  assert "if(ACTIVE==='velocity')return saveVelocityPng();" in content
  assert "if(key==='velocity')" in content


def test_tn_joint_ratios_use_full_selected_samples_and_export_them() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function tnJointRatioRows(joint,start,end)" in content
  assert "stats:tnSummaryStats(start,end,[joint])" in content
  assert "tnSummaryStats(start,end,[joint],[side])" in content
  assert "zeroPct:100*stats.zero/stats.total" in content
  assert "card.insertAdjacentHTML('beforeend',tnJointSummaryHtml(joint));" in content
  assert "text:tnJointRatioText(joint)" in content
  assert "html+=tnJointSummaryHtml(joint);" in content
  assert "工况筛选仅过滤散点" in content


def test_world_position_is_an_xy_time_colored_trajectory_without_boxes() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function velocityPositionOption(" in content
  assert "const point=[x[i],y[i],DATA.time[i]]" in content
  assert "dimension:2,seriesIndex:0" in content
  assert "name:'x (m)'" in content
  assert "name:'y (m)'" in content
  assert "return group.id!=='standing'&&group.id!=='position';" in content
  assert content.count("if(rows.length&&velocityWantsBox(group))") == 2
  assert "seriesIndex:series.length" in content


def test_foot_layout_wraps_notes_and_adapts_grid_columns() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function footChartCard(el)" in content
  assert "function footUpdatePresentation(chart,el)" in content
  assert "repeat(auto-fit,minmax(min(100%,440px),1fr))" in content
  assert "overflow-wrap:anywhere" in content
  assert "replaceMerge:['graphic']" in content
  assert "if(FOOT_ZOOM?.el===el)return FOOT_ZOOM.parent;" in content


def test_step_boxes_use_independent_unit_grids_and_real_sample_counts() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "const units=[...new Set(rows.map(([name])=>nameUnit[name]))];" in content
  assert "s.xAxisIndex=gi;s.yAxisIndex=gi;" in content
  assert "n=${nameN[name]} 步" in content
  assert "混合量纲，见 tooltip" not in content


def test_step_statistics_exclude_clipped_stances_and_keep_equal_duties() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function footCompleteStances(side,start,end)" in content
  assert "a>=start&&a>0&&b<=end&&b<c.length" in content
  assert "footTouchdowns(side,start,end).map(e=>arr[e.idx])" in content
  assert "for(const [i0,i1] of footCompleteStances(side,start,end))" in content
  assert "const spans=footCompleteStances(side,start,end);" in content
  assert "if(Number.isFinite(contact[i])){n++;" in content


def test_area_tooltip_selects_side_by_series_index() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  chart = content[content.index("function renderFootAreaChart(el)") :]
  chart = chart[: chart.index("function renderFootPressureTime(el)")]

  assert "const side=q.seriesIndex===0?'left':'right';" in chart
  assert "footSeries('contact_fraction',side)?.[i]" in chart


def test_foot_weight_reference_requires_explicit_whole_robot_mass() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  helper = content[content.index("function footForceMgRef(start,end)") :]
  helper = helper[: helper.index("function footImpactEvents")]

  assert 'id="foot-mass-kg"' in content
  assert "document.getElementById('foot-mass-kg')?.value" in helper
  assert "source:'physical'" in helper
  assert "区间总法向力均值（非 mg）" in helper
  assert "6.76" not in helper
  assert "不能据此判断训练或 reset 时是否开启 DR" in content


def test_cop_uses_time_colors_and_does_not_claim_nominal_hull_is_support() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "gy.push((Math.floor(j/5)-2)*14);" in content
  assert "dots.push({value:[...mm,DATA.time[i]],idx:i});" in content
  assert "名义网格框（非接触凸包）" in content
  assert "if(el.clientWidth<620)" in content
  assert "formatter:v=>String(Number(v.toPrecision(4)))" in content


def test_foot_pdf_uses_specialized_charts_without_changing_live_view() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  pdf = content[content.index("async function buildAllPdf()") :]

  assert "function footPdfHtml(key)" in content
  assert "html+=footPdfHtml(key);" in pdf
  assert pdf.index("if(FOOT_CATEGORIES.has(key))") < pdf.index(
    "overviewOption(names,true,key)"
  )
  assert (
    "finally{ACTIVE=previous;FOOT_PRESSURE_VIEW=pressureView;FOOT_RENDER_STATIC=staticRender;host.remove();}"
    in content
  )
  assert "finally{chart?.dispose();}" in content


def test_foot_empty_ranges_skip_blank_cards_but_keep_finite_zeros() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert ".slice(start,end).some(Number.isFinite)" in content
  assert "footStepValues(stem,side,start,end).length>0" in content
  assert 'id="foot-other-table"' in content


def test_foot_playback_is_shared_across_subgroups_and_pressure() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  assert 'id="foot-timeline-controls"' in content
  assert "function ensureFootPlayback()" in content
  assert "function drawFootFrame(index)" in content
  assert "mountFootPlaybackControls(timeline);" in content
  assert content.count("mountFootPlaybackControls(el,chart);") == 2
  assert "FOOT_PRESSURE_CURSOR_DATA!==DATA" in content
  assert "player.speed=FOOT_PLAYBACK_SPEED" in content


def test_foot_playback_does_not_interpolate_or_autoplay_invalid_time() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  assert "time[start+i-1]" in content
  assert "envs.size>1?'多环境混合日志" in content
  assert "player.data.time[index+1]<=target" in content
  assert "Math.floor(elapsed*25)" in content
  assert "if(!player||player.reason)return;" in content


def test_foot_frames_show_raw_values_and_time_cursors_not_distributions() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  assert "for(const name of CATEGORIES[ACTIVE]||[])" in content
  assert "node.textContent=name+' = '" in content
  assert "function attachFootTimeCursor(chart,el)" in content
  assert "name:'当前帧',xAxis:DATA.time[index]" in content
  assert "[FOOT_STEPBOX_ID,'foot-cop-chart'" in content
  assert "if(!axes.length||axes.some(axis=>axis.type!=='value'))return;" in content


def test_force_snapshots_keep_magnitudes_and_missing_values() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  force = content.split("function renderFootForceFrame(el){")[1].split(
    "function renderFootForceChart(el){"
  )[0]
  assert "['法向接触力','切向力大小']" in force
  assert "return Number.isFinite(value)?value:null;" in force
  assert "values.reduce((a,b)=>Math.max(a,b),1)*1.08" in force
  assert "type:'bar'" in force
  assert "不虚构方向" in force


def test_cop_frame_is_oriented_like_pressure_with_current_marker() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  frame = content.split("function renderFootCopFrame(el){")[1].split(
    "function renderFootCopOverview(el){"
  )[0]
  assert "return [1,0].map(i=>" in frame
  assert "[-y[index]*1e3,x[index]*1e3]" in frame
  assert "const height=gridHeight(),width=height*aspect;" in frame
  assert "api.size([14,20])" in frame
  assert "无压力阵列，仅显示 CoP" in frame


def test_cop_trails_do_not_show_future_airborne_or_missing_samples() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  assert "gaitContactState(footSeries('contact',side)?.[i])!==0" in content
  assert "Number.isFinite(x?.[i])&&Number.isFinite(y?.[i])" in content
  assert "i=index;i>=start&&i>=index-49&&valid(i);i--" in content
  assert "DATA.time[index]-DATA.time[i]>1" in content
  assert "let FOOT_COP_VIEW='time';" in content


def test_foot_playback_cleanup_and_pdf_isolation() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  assert "FOOT_FRAME_RENDERERS.clear();FOOT_FRAME_READOUTS=[];" in content
  assert "cancelAnimationFrame(player.raf)" in content
  assert "if(FOOT_RENDER_STATIC)return;" in content
  assert "FOOT_RENDER_STATIC||FOOT_COP_VIEW==='trajectory'" in content
  assert "FOOT_RENDER_STATIC=staticRender;host.remove();" in content
  assert "未登记字段：保留原始统计，不猜测单位或跨单位叠图。" in content
  assert "if(!isFootCategory)footDisposeAll();" in content
  assert "out.push(n?(m==='peak'?mx:sum/n):NaN);" in content


def test_motor_selection_table_uses_real_joint_sides_and_has_controls() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert 'id="motor-config-panel"' in content
  assert 'id="open-motor-config"' in content
  assert "function motorConfigNames()" in content
  assert "meta.side&&meta.joint&&meta.category.startsWith('joint_')" in content
  assert 'data-motor-field="type"' in content
  assert 'data-motor-field="ratio"' in content
  assert "function copyMotorSide(side)" in content


def test_motor_selection_is_explicitly_an_extra_ratio_not_total_gearing() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "const MOTOR_RATIO_CONVENTION='reference-output-to-joint-extra';" in content
  assert "不能重复计入电机内置减速器" in content
  assert "这不是换电机后的动力学仿真" in content
  assert "record.ratio<0.001||record.ratio>1000" in content


def test_motor_selection_survives_reload_and_has_versioned_json_io() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "let MOTOR_OVERRIDES=loadMotorOverrides();" in content
  assert "localStorage.setItem(MOTOR_CONFIG_STORAGE_KEY" in content
  assert "data.version!==1||data.ratioConvention!==MOTOR_RATIO_CONVENTION" in content
  assert (
    "const motors=parseMotorConfig(await file.text());if(!allowMotorConfigEdit())return;MOTOR_OVERRIDES=motors;"
    in content
  )
  assert "原配置保持不变" in content
  assert "'motor-selection.json'" in content


def test_manual_motor_selection_takes_precedence_over_xml_defaults() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  helper = content[content.index("function motorConfigFor(joint)") :]
  helper = helper[: helper.index("function motorTypeFor")]

  assert "const record=MOTOR_OVERRIDES[joint];" in helper
  assert "return record?{...record}:{type:motorBaseTypeFor(joint),ratio:1};" in helper
  assert "保留手动选型覆盖" in content
  assert "function resetMotorOverrides()" in content


def test_motor_tn_transforms_speed_and_torque_without_transforming_samples() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function tnMotorSeriesData(mode,uRpm,swap,mirror=false,ratio=1)" in content
  assert "point.speedRpm*2*Math.PI/60)/ratio" in content
  assert "const torque=sign*point.torqueNm*ratio;" in content
  assert "const key=mode+':'+c.ratio;" in content
  assert "rawSpeed:v,rawTorque:t,powerW:power" in content


def test_motor_rated_lines_distinguish_same_model_with_different_ratios() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "return [c.type+':'+c.ratio,c];" in content
  assert "const value=sign*MOTOR_RATED_TORQUE[type]*ratio" in content
  assert "function ratedTorqueFor(joint)" in content


def test_motor_selection_updates_comparisons_and_export_metadata() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "tnMotorReferenceSeries([metric.slice('joint_tn/'.length)]" in content
  assert "CMP.metrics.some(isTnCompareMetric)&&TN_REF!=='none'" in content
  assert "function motorConfigPrintHtml()" in content
  assert "row.deleteCell(row.cells.length-1)" in content
  assert "候选电机额外减速比：理想传动" in content
  assert "s.data.map(p=>(p.value||p)[0])" in content


def test_motor_selection_electrical_estimate_and_inertia_reference_use_ratio() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "ratio=motorConfigFor(tag).ratio,I=Math.abs(τ)/(mp.kt*ratio)" in content
  assert "st.peak>ratedTorqueFor(`${side}_${joint}_joint`)" in content
  assert "旧参考Torq限" in content
  assert "候选额定 N·m" in content
  assert "瞬时超过额定不等于超过峰值能力" in content


def test_motor_config_is_locked_during_exports_and_curves_are_awaited() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function allowMotorConfigEdit()" in content
  assert "if(TN_PNG_EXPORTING||PDF_EXPORTING)" in content
  assert "if(!allowMotorConfigEdit())return;" in content
  assert (
    content.count(
      "if(TN_OPERATION!=='brake')await Promise.all(tnReferenceModes(TN_REF).map(loadTnMotorCurve));"
    )
    == 2
  )


def test_gait_overlay_has_a_sidebar_toggle_independent_of_curve_selection() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert 'id="gait-overlay-toggle"' in content
  assert 'id="gait-overlay-status"' in content
  assert content.index('id="gait-overlay-toggle"') < content.index(
    'id="aside-term-controls"'
  )
  assert "let GAIT_OVERLAY_ON = false;" in content
  assert "button.setAttribute('aria-pressed'" in content


def test_gait_overlay_uses_strict_contact_flags_and_guards_multiple_envs() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function gaitContactState(value)" in content
  assert "if(!Number.isFinite(value))return null;" in content
  assert "Math.abs(value-1)<=1e-6?1:null" in content
  assert "if(envs.size>1)return {sides:[]" in content
  assert "不用相位或力值猜测" in content


def test_gait_overlay_clips_spans_without_extrapolating_final_sample() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function gaitOverlaySpans(" in content
  assert "toX(Math.min(stop,end-1))" in content
  assert "if(Number.isFinite(a)&&Number.isFinite(b)&&b>a)" in content
  assert "if(onset!==null)append(end-1);" in content


def test_gait_overlay_adds_backgrounds_not_extra_measurement_series() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")
  helper = content[content.index("function withGaitOverlay(") :]
  helper = helper[: helper.index("function syncGaitOverlayUi()")]

  assert "const series=(option.series||[]).map" in helper
  assert "item.markArea={...item.markArea,silent:true,z:0" in helper
  assert "return {...option,series};" in helper
  assert "series.push(" not in helper
  assert "DATA.values[" not in helper


def test_gait_overlay_covers_time_factories_without_xy_or_tn_inference() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "return withGaitOverlay({" in content
  assert "POWER_CHART.setOption(withGaitOverlay({" in content
  assert "if(group.id==='position')return velocityPositionOption" in content
  assert "patch.series=withGaitOverlay({xAxis:timeAxes" in content
  spatial = content[content.index("function velocityPositionOption(") :]
  spatial = spatial[: spatial.index("function velocityWantsBox(")]
  assert "withGaitOverlay" not in spatial


def test_gait_overlay_is_idempotent_and_preserves_non_gait_reference_bands() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "!area[0]?.__gaitOverlay&&!area[0]?.__gaitContact" in content
  assert "__gaitContact:true" in content
  assert "__gaitOverlay:true" in content
  assert "足端原有接触提示保留" in content


def test_comparison_gait_uses_each_source_and_the_matching_x_transform() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "const gaitContexts=[L,R].map((dataset,i)" in content
  assert "seriesIndex:i" in content
  assert "index/(dataset.time.length-1)*100" in content
  assert "dataset.time[index]-dataset.time[0]" in content
  assert "},gaitContexts));" in content
  assert "borderType:sourceIndex===1?'dashed':'solid'" in content


def test_gait_overlay_exports_retain_a_legend_and_lock_toggle_during_export() -> None:
  content = DASHBOARD.read_text(encoding="utf-8")

  assert "function gaitOverlayCaption()" in content
  assert "步态背景：浅蓝=左脚触地" in content
  assert "正在导出，完成后再切换步态叠加" in content
  assert "graphic:gaitOverlayCaption()?" in content
