# ============================================================
# 氢电智算 — 全功率燃料电池仿真计算APP（网页版）
# 单文件版，包含后端API + 前端网页
# ============================================================

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="氢电智算")

# ============ 数据模型 ============
class HydrogenReq(BaseModel):
    power_kw: float
    efficiency: float
    lhv: float = 120.0

class AirReq(BaseModel):
    power_kw: float
    efficiency: float
    lam: float = 2.0

class ThermalReq(BaseModel):
    power_kw: float
    efficiency: float
    dt: float = 10.0

class SimReq(BaseModel):
    sim_type: str
    power_kw: float = 120.0

# ============ 核心计算引擎 ============
def polarization_curve(i_max=2.0, n=80, e0=1.23, a=0.06, i0=0.001,
                       r_ohm=0.15, b=0.08, i_lim=2.2, temp=80, press=2.0):
    i = np.linspace(0.001, i_max, n)
    e = e0 + 0.00085*(temp-25) + 0.0125*np.log(press)
    eta_act = a*np.log(i/i0)
    eta_ohm = i*r_ohm
    eta_conc = -b*np.log(1-i/i_lim)
    v = np.clip(e - eta_act - eta_ohm - eta_conc, 0, None)
    pd = v * i
    return {
        "current_density": i.tolist(),
        "voltage": v.tolist(),
        "power_density": pd.tolist(),
        "peak_power_density": float(np.max(pd)),
        "losses": {
            "activation": eta_act.tolist(),
            "ohmic": eta_ohm.tolist(),
            "concentration": eta_conc.tolist()
        }
    }

# ============ API 接口 ============
@app.post("/api/hydrogen")
def calc_hydrogen(req: HydrogenReq):
    power_mj_s = req.power_kw * 0.001
    mass_flow = power_mj_s / (req.lhv * req.efficiency) * 3600
    vol_flow = mass_flow * 11.126
    heat_input = req.power_kw / req.efficiency
    return {
        "mass_flow_kg_h": round(mass_flow, 4),
        "vol_flow_nm3_h": round(vol_flow, 2),
        "heat_input_kw": round(heat_input, 2)
    }

@app.post("/api/air")
def calc_air(req: AirReq):
    theo_o2 = req.power_kw * 0.2696 / req.efficiency
    theo_air = theo_o2 / 0.232
    actual_air = theo_air * req.lam
    compressor = actual_air * 0.030
    return {
        "theoretical_air_kg_h": round(theo_air, 2),
        "actual_air_kg_h": round(actual_air, 2),
        "compressor_power_kw": round(compressor, 3)
    }

@app.post("/api/thermal")
def calc_thermal(req: ThermalReq):
    heat_kw = req.power_kw * (1/req.efficiency - 1)
    flow = heat_kw / (4.18 * req.dt)
    return {"heat_kw": round(heat_kw, 2), "coolant_flow_kg_s": round(flow, 4)}

@app.post("/api/simulate")
def simulate(req: SimReq):
    t = req.sim_type
    warnings = []
    if t == "steady":
        c = polarization_curve(i_max=1.5)
    elif t == "dynamic":
        c = polarization_curve(i_max=2.0)
        warnings.append("动态变载：建议空压机响应时间 < 200ms")
    elif t == "cold_start":
        c = polarization_curve(temp=-20)
        warnings.append("低温冷启动：需启用外部加热或催化燃烧预热")
    elif t == "limit":
        c = polarization_curve(i_max=2.5, i_lim=2.6)
        warnings.append("边界工况：电流密度超过 2.0 A/cm² 存在碳腐蚀风险")
    else:
        c = polarization_curve()
    return {
        "curves": c,
        "summary": {
            "peak_power_density": c["peak_power_density"],
            "power_kw": req.power_kw,
            "efficiency_estimate": 0.55
        },
        "warnings": warnings
    }

@app.post("/api/diagnose")
def diagnose(data: dict):
    symptom = data.get("symptom", "")
    power = data.get("power_kw", 120)
    # 简易规则诊断
    if "水淹" in symptom or "电压低" in symptom:
        result = f"""【诊断结果】{power}kW 系统疑似水淹
【故障位置】电堆本体 / 阴极水管理
【故障树推理】
  · 电流密度过高 → 生成水速率 > 排出速率
  · 冷却液温度过低 → 饱和蒸汽压下降
  · 空气流量不足 → 吹扫能力不够
【优化建议】
  1. 提升冷却液温度 3~5℃
  2. 增加空气过量系数 λ 至 2.2
  3. 降低电流密度至 1.2 A/cm² 以下运行
  4. 检查背压阀开度"""
    elif "膜干" in symptom or "阻抗大" in symptom:
        result = f"""【诊断结果】{power}kW 系统疑似膜干
【故障位置】电堆本体 / 水热管理
【故障树推理】
  · 温度过高 → 膜内水蒸发
  · 增湿不足 → 进气湿度偏低
  · 电流密度过高 → 电渗拖拽加剧
【优化建议】
  1. 降低冷却液温度 3~5℃
  2. 提高增湿器温度
  3. 提高进气相对湿度到 80% 以上"""
    elif "一致性" in symptom or "离散" in symptom:
        result = f"""【诊断结果】{power}kW 系统单体一致性差
【故障位置】电堆本体 / 分配歧管
【故障树推理】
  · 分配歧管流阻不均
  · 局部水淹或膜干
  · 单片膜电极老化
【优化建议】
  1. 检查分配歧管压降
  2. 做单体电压离散步分析
  3. 排查异常单片的MEA状态"""
    else:
        result = f"""【诊断结果】{power}kW 系统综合诊断
【建议】
  1. 请提供更详细的故障现象（电压、温度、阻抗等）
  2. 或进行极化曲线测试，进行三损失分解分析
  3. 可上传 EIS 数据进行 Randles 等效电路拟合"""
    return {"diagnosis": result}

# ============ 前端网页 ============
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>氢电智算</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
body { font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
       background:#0a1929; color:#e6f1ff; min-height:100vh; padding-bottom:80px; }
.header { background:linear-gradient(135deg,#0077C8,#00A6FB); padding:20px;
          text-align:center; position:sticky; top:0; z-index:10; box-shadow:0 2px 12px rgba(0,166,251,.3); }
.header h1 { font-size:22px; letter-spacing:2px; }
.header p { font-size:12px; opacity:.9; margin-top:4px; }
.container { padding:16px; max-width:800px; margin:0 auto; }
.card { background:rgba(255,255,255,.06); border:1px solid rgba(0,166,251,.2);
        border-radius:12px; padding:16px; margin-bottom:14px; backdrop-filter:blur(10px); }
.card h3 { color:#00A6FB; font-size:15px; margin-bottom:12px; display:flex; align-items:center; gap:8px; }
label { display:block; font-size:13px; color:#8aa6c1; margin:10px 0 4px; }
input, select { width:100%; padding:11px 12px; background:rgba(0,0,0,.3);
       border:1px solid rgba(0,166,251,.3); border-radius:8px; color:#e6f1ff;
       font-size:15px; outline:none; transition:.2s; }
input:focus, select:focus { border-color:#00A6FB; box-shadow:0 0 0 3px rgba(0,166,251,.15); }
button { width:100%; padding:13px; background:linear-gradient(135deg,#0077C8,#00A6FB);
         border:none; border-radius:8px; color:#fff; font-size:15px; font-weight:600;
         margin-top:14px; cursor:pointer; transition:.2s; }
button:active { transform:scale(.98); }
button:disabled { opacity:.6; }
.result { background:rgba(0,166,251,.1); border-left:3px solid #00A6FB;
          padding:12px; border-radius:6px; margin-top:14px; font-size:13px;
          line-height:1.9; white-space:pre-wrap; word-break:break-all; }
.result:empty { display:none; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
.chip { padding:7px 14px; background:rgba(0,166,251,.1); border:1px solid rgba(0,166,251,.3);
        border-radius:20px; font-size:12px; cursor:pointer; transition:.2s; }
.chip.active { background:#00A6FB; color:#fff; border-color:#00A6FB; }
.nav { position:fixed; bottom:0; left:0; right:0; background:#0d2137;
       display:flex; border-top:1px solid rgba(0,166,251,.2); z-index:20; }
.nav-item { flex:1; text-align:center; padding:10px 4px; font-size:11px;
            color:#8aa6c1; cursor:pointer; transition:.2s; }
.nav-item.active { color:#00A6FB; }
.nav-item .ico { font-size:20px; display:block; margin-bottom:2px; }
.page { display:none; }
.page.active { display:block; }
.warn { background:rgba(255,152,0,.15); border-left:3px solid #FF9800;
        padding:10px; border-radius:6px; margin-top:10px; font-size:12px; line-height:1.7; }
canvas { max-height:280px; }
@media(max-width:480px){ .container{padding:12px;} .card{padding:14px;} }
</style>
</head>
<body>

<div class="header">
  <h1>⚡ 氢电智算</h1>
  <p>30kW~400kW 燃料电池仿真计算</p>
</div>

<div class="container">

  <!-- 首页 -->
  <div class="page active" id="page-home">
    <div class="card">
      <h3> 快速开始</h3>
      <p style="font-size:13px;line-height:1.8;color:#8aa6c1;">
        覆盖 30kW~400kW 燃料电池系统与测试台架<br>
        参数管理 · 工况仿真 · 工程计算 · 智能诊断
      </p>
    </div>
    <div class="card">
      <h3> 功率分段</h3>
      <div class="result" style="margin:0;background:transparent;border:none;padding:0;">
小功率 30~80kW ｜ 乘用车 / 轻型物流车
中功率 100~200kW ｜ 客车 / 重卡 / 工业发电
大功率 250~400kW ｜ 重型装备 / 大型电站
      </div>
    </div>
    <div class="card">
      <h3> 六大仿真模式</h3>
      <div class="chips">
        <div class="chip">稳态额定</div>
        <div class="chip">动态变载</div>
        <div class="chip">低温冷启动</div>
        <div class="chip">耐久循环</div>
        <div class="chip">台架对标</div>
        <div class="chip">极限边界</div>
      </div>
    </div>
  </div>

  <!-- 计算器 -->
  <div class="page" id="page-calc">
    <div class="card">
      <h3> 工程快捷计算</h3>
      <div class="chips" id="calcTabs">
        <div class="chip active" data-tab="h2">氢气消耗</div>
        <div class="chip" data-tab="air">空气流量</div>
        <div class="chip" data-tab="heat">热管理</div>
      </div>

      <div id="calc-h2">
        <label>电堆功率 (kW)</label>
        <input type="number" id="h2-p" value="120">
        <label>电堆效率 (%)</label>
        <input type="number" id="h2-e" value="55">
        <button onclick="calcH2()">开始计算</button>
        <div class="result" id="h2-r"></div>
      </div>

      <div id="calc-air" style="display:none">
        <label>电堆功率 (kW)</label>
        <input type="number" id="air-p" value="120">
        <label>电堆效率 (%)</label>
        <input type="number" id="air-e" value="55">
        <label>空气过量系数 λ</label>
        <input type="number" id="air-l" value="2.0" step="0.1">
        <button onclick="calcAir()">开始计算</button>
        <div class="result" id="air-r"></div>
      </div>

      <div id="calc-heat" style="display:none">
        <label>电堆功率 (kW)</label>
        <input type="number" id="heat-p" value="120">
        <label>电堆效率 (%)</label>
        <input type="number" id="heat-e" value="55">
        <label>冷却液温差 ΔT (K)</label>
        <input type="number" id="heat-dt" value="10">
        <button onclick="calcHeat()">开始计算</button>
        <div class="result" id="heat-r"></div>
      </div>
    </div>
  </div>

  <!-- 仿真 -->
  <div class="page" id="page-sim">
    <div class="card">
      <h3> 全工况仿真</h3>
      <div class="chips" id="simTabs">
        <div class="chip active" data-t="steady">稳态额定</div>
        <div class="chip" data-t="dynamic">动态变载</div>
        <div class="chip" data-t="cold_start">低温冷启动</div>
        <div class="chip" data-t="limit">极限边界</div>
      </div>
      <label>系统功率 (kW)</label>
      <input type="number" id="sim-p" value="120">
      <button onclick="runSim()" id="simBtn">运行仿真</button>
      <div class="result" id="sim-r"></div>
      <div class="warn" id="sim-warn" style="display:none"></div>
    </div>
    <div class="card" id="chartCard" style="display:none">
      <h3> 极化曲线</h3>
      <canvas id="polarChart"></canvas>
    </div>
  </div>

  <!-- 诊断 -->
  <div class="page" id="page-diag">
    <div class="card">
      <h3>喙 智能故障诊断</h3>
      <label>系统功率 (kW)</label>
      <input type="number" id="diag-p" value="120">
      <label>描述故障现象</label>
      <input type="text" id="diag-s" placeholder="如：电压低、水淹、一致性差">
      <div class="chips" style="margin-top:10px;">
        <div class="chip" onclick="fillDiag('部分单体电压低于0.5V，堆温异常，怀疑水淹')">示例：水淹</div>
        <div class="chip" onclick="fillDiag('高频阻抗增大，膜干，温度偏高')">示例：膜干</div>
        <div class="chip" onclick="fillDiag('单体电压离散度增大，一致性差')">示例：一致性差</div>
      </div>
      <button onclick="runDiag()">开始诊断</button>
      <div class="result" id="diag-r"></div>
    </div>
  </div>

  <!-- 知识库 -->
  <div class="page" id="page-kb">
    <div class="card">
      <h3> 国标规范</h3>
      <div class="result" style="margin:0;background:transparent;border:none;padding:0;font-size:13px;">
        · GB/T 24549 燃料电池电动汽车 安全要求<br>
        · GB/T 28816 燃料电池 术语<br>
        · GB/T 33978 道路车辆用质子交换膜燃料电池模块
      </div>
    </div>
    <div class="card">
      <h3> 测试规范</h3>
      <div class="result" style="margin:0;background:transparent;border:none;padding:0;font-size:13px;">
        · 极化曲线：稳态 I-V 曲线测试<br>
        · EIS：10mHz~10kHz，交流幅值 5%<br>
        · 冷启动：-30℃ 启动时间 &lt; 30s
      </div>
    </div>
    <div class="card">
      <h3>⚠️ 故障机理</h3>
      <div class="result" style="margin:0;background:transparent;border:none;padding:0;font-size:13px;">
        · 水淹：高电流密度 + 低温 + 空气不足<br>
        · 膜干：高温 + 增湿不足 + 高电流密度<br>
        · 碳腐蚀：高电位 + 频繁启停 + 低湿度
      </div>
    </div>
  </div>

</div>

<div class="nav">
  <div class="nav-item active" onclick="go('home',this)"><span class="ico"></span>首页</div>
  <div class="nav-item" onclick="go('calc',this)"><span class="ico"></span>计算</div>
  <div class="nav-item" onclick="go('sim',this)"><span class="ico"></span>仿真</div>
  <div class="nav-item" onclick="go('diag',this)"><span class="ico">喙</span>诊断</div>
  <div class="nav-item" onclick="go('kb',this)"><span class="ico"></span>知识库</div>
</div>

<script>
function go(id, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
  el.classList.add('active');
  window.scrollTo(0,0);
}

// 计算器 Tab 切换
document.querySelectorAll('#calcTabs .chip').forEach(c => {
  c.onclick = () => {
    document.querySelectorAll('#calcTabs .chip').forEach(x => x.classList.remove('active'));
    c.classList.add('active');
    document.querySelectorAll('[id^=calc-]').forEach(p => p.style.display='none');
    document.getElementById('calc-' + c.dataset.tab).style.display='block';
  };
});

// 仿真 Tab 切换
let simType = 'steady';
document.querySelectorAll('#simTabs .chip').forEach(c => {
  c.onclick = () => {
    document.querySelectorAll('#simTabs .chip').forEach(x => x.classList.remove('active'));
    c.classList.add('active');
    simType = c.dataset.t;
  };
});

async function api(path, data) {
  const r = await fetch('/api/' + path, {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)
  });
  return await r.json();
}

async function calcH2() {
  const p = +document.getElementById('h2-p').value;
  const e = +document.getElementById('h2-e').value / 100;
  const r = await api('hydrogen', {power_kw:p, efficiency:e, lhv:120});
  document.getElementById('h2-r').textContent =
    `质量流量：${r.mass_flow_kg_h} kg/h\\n体积流量：${r.vol_flow_nm3_h} Nm³/h\\n热输入功率：${r.heat_input_kw} kW`;
}

async function calcAir() {
  const p = +document.getElementById('air-p').value;
  const e = +document.getElementById('air-e').value / 100;
  const l = +document.getElementById('air-l').value;
  const r = await api('air', {power_kw:p, efficiency:e, lam:l});
  document.getElementById('air-r').textContent =
    `理论空气量：${r.theoretical_air_kg_h} kg/h\\n实际供气量：${r.actual_air_kg_h} kg/h\\n空压机功耗：${r.compressor_power_kw} kW`;
}

async function calcHeat() {
  const p = +document.getElementById('heat-p').value;
  const e = +document.getElementById('heat-e').value / 100;
  const dt = +document.getElementById('heat-dt').value;
  const r = await api('thermal', {power_kw:p, efficiency:e, dt:dt});
  document.getElementById('heat-r').textContent =
    `产热量：${r.heat_kw} kW\\n冷却液流量：${r.coolant_flow_kg_s} kg/s`;
}

let chart;
async function runSim() {
  const btn = document.getElementById('simBtn');
  btn.disabled = true; btn.textContent = '仿真中...';
  const p = +document.getElementById('sim-p').value;
  const r = await api('simulate', {sim_type:simType, power_kw:p});
  btn.disabled = false; btn.textContent = '运行仿真';

  document.getElementById('sim-r').textContent =
    `峰值功率密度：${r.summary.peak_power_density.toFixed(3)} W/cm²\\n系统功率：${r.summary.power_kw} kW\\n估算效率：${(r.summary.efficiency_estimate*100).toFixed(1)}%`;

  const warnBox = document.getElementById('sim-warn');
  if (r.warnings.length) { warnBox.style.display='block'; warnBox.textContent = '⚠️ ' + r.warnings.join('\\n⚠️ '); }
  else warnBox.style.display='none';

  document.getElementById('chartCard').style.display='block';
  const cd = r.curves.current_density;
  const v = r.curves.voltage;
  const pd = r.curves.power_density;

  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('polarChart'), {
    type:'line',
    data:{ labels: cd.map(x=>x.toFixed(2)),
      datasets:[
        { label:'电压 (V)', data:v, borderColor:'#00A6FB', tension:.3, pointRadius:0, yAxisID:'y' },
        { label:'功率密度 (W/cm²)', data:pd, borderColor:'#FF6B6B', tension:.3, pointRadius:0, yAxisID:'y1' }
      ]},
    options:{ responsive:true, plugins:{legend:{labels:{color:'#e6f1ff'}}},
      scales:{
        x:{ ticks:{color:'#8aa6c1'}, grid:{color:'rgba(255,255,255,.05)'} },
        y:{ position:'left', ticks:{color:'#00A6FB'}, grid:{color:'rgba(255,255,255,.05)'}, title:{display:true,text:'电压 V',color:'#00A6FB'} },
        y1:{ position:'right', ticks:{color:'#FF6B6B'}, grid:{display:false}, title:{display:true,text:'功率 W/cm²',color:'#FF6B6B'} }
      }}
  });
}

function fillDiag(t){ document.getElementById('diag-s').value = t; }

async function runDiag() {
  const p = +document.getElementById('diag-p').value;
  const s = document.getElementById('diag-s').value;
  if (!s) { alert('请输入故障现象'); return; }
  document.getElementById('diag-r').textContent = 'AI 分析中...';
  const r = await api('diagnose', {symptom:s, power_kw:p});
  document.getElementById('diag-r').textContent = r.diagnosis;
}
</script>
</body>
</html>
"""
