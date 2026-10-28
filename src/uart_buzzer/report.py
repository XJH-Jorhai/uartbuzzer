from pathlib import Path
import json
import numpy as np


def make_report(root,out,p,model,fitted,ac,rows,checkpoints):
    from .supplements import build
    build(root,out,p)
    def relative(path):return Path(path).relative_to(out).as_posix()
    def table(cols,rs):
        return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(str(r.get(k,'')) for k in cols)+' |' for r in rs)+'\n'
    def fmt(rs):
        return [{k:(f'{v:.6g}' if isinstance(v,float) else v) for k,v in r.items()} for r in rs]
    names=['Rcu (ohm)','Linf (H)','Reddy (ohm)','Leddy (H)','Rmot (ohm)','Lmot (H)','Cmot (F)','Rmot2 (ohm)','Lmot2 (H)','Cmot2 (F)']
    discovery=json.loads((out/'discovery.json').read_text(encoding='utf-8'))
    findings=''
    uartfile=out/'tables/uart_comparison.csv'
    if uartfile.exists():
        from .io_data import read_csv
        uart=read_csv(uartfile)
        chosen=[r for r in uart if int(r['f_Hz'])==2000 and int(r['baud']) in (460800,921600)]
        fields=['topology','baud','method','fundamental_peak_A','THD_2to5','inband_residual_to_fundamental','Pcu_W','missing_edges','delay_max_s']
        findings+='\n### 2kHz 编码器实算比较\n\n'+table(fields,[{k:(f'{float(v):.6g}' if k in fields[3:] else v) for k,v in r.items() if k in fields} for r in chosen])
        findings+='\n![线圈电流频谱](plots/uart_spectra.png)\n\n![跨音域与波特率比较](plots/tone_comparison.png)\n'
    worst=[r for r in rows if r['case'].endswith('_sampled_worst')]
    worst_text=''
    if worst:
        worst_text=table(['case','Ipeak_A','Irms_A','Pcu_W','Vswitch_abs_peak_V','Vcontrol_abs_peak_V','TX_sink_peak_A','Plimit_W'],fmt(worst))
        worst_text+='\n![PNP 已采样最坏角点](plots/pnp_sampled_worst.png)\n\n![PMOS 已采样最坏角点](plots/pmos_sampled_worst.png)\n'
    edge_scan=''
    scanfile=out/'tables/component_scan.csv'
    if scanfile.exists():
        from .io_data import read_csv
        scans=read_csv(scanfile)
        selected=[r for r in scans if (r['case'].startswith('gate_') and float(r['rgs'])==10000)
                    or (r['case'].startswith('base_') and float(r['rpull'])==4700)]
        edge_scan=table(['case','TX_sink_peak_A','TX_source_peak_A','delay_max_s','missing_edges','delivered_ON_density'],selected)
    pwl_note='UART激励PWL时刻严格递增，无冗余初始角点。'
    if any('non-increasing PWL' in q.read_text(encoding='utf-8') for q in (out/'spice').glob('*_2000.log')):
        pwl_note='本轮原始UART网表日志保留了初始空闲与首个边沿处重复同电压时刻的PWL警告。交付代码会去掉该冗余角点；补充复核 pwl_cleanup_verification.json 比较清理前后电流，并检查新网表时刻严格递增。原始日志未被改写。'
    text=f'''# CH343 UART-PDM 电磁蜂鸣器可复现研究

运行目录：`{out}`。阶段状态：{checkpoints}。本报告来自实际 Python/ngspice 运行，共 {len(rows)} 个驱动瞬态案例。它是受模型限制的电气可行性研究，硬件最终认证、绝对 SPL 和安全温升均为 **unverified / 未知**。

## 输入证据与事实等级

- 测量：原始 TH2851 宽扫、精扫、DCR 和低电平检查，读取 utf-8-sig，原始列不改动；只读使用原件，并保留按 SHA256 命名的副本。完整清单与哈希见 [input_manifest.json](input_manifest.json)。
- 文件按字段递归发现，实际选用：{discovery['selected']}。指定别名未出现时使用已有等价字段文件，未伪造说明文件。
- 缺少的说明文件：{discovery['missing']}。仪器设置和校准未获得这两份文件的独立复核。现有日志/字段能证明的是记录内容，不能补造校准证书。
- DLOG 状态见 [dlog_status.json](dlog_status.json)。未找到则无法解析、绘图或验证采样率；约 24.414 kSa/s、25.4 µF 均仅为任务书背景假设。任何端口电流都不等于线圈电流，不作 RMS 拟合基准。
- 仿真：ngspice 计算的非线性晶体管、二极管和暂态网络响应；不称为实测。拟合是 100 mV 设定、无偏置下的输入阻抗。VAC/IAC 可疑交换且可能推算，不作为独立幅度校准；质量交叉检查见 [data_quality.json](data_quality.json)。
- 推断：5 V 大信号、UART 高频、热模型、机械运动代理、器件典型模型的外推；没有磁饱和/电感随电流/机械非线性的测量支持。

5mV 与100mV复阻抗交叉检查见 [low_level_uncertainty.csv](tables/low_level_uncertainty.csv)。它混合了激励、不同时间/重复性和温漂影响，不能单独证明激励非线性。输入完整性再次核对见 input_integrity.json；实际引擎为 ngspice-46，版本输出见 ngspice_version.log。21项测试（含重复拟合、M2能量和真实ngspice）见 [tests.log](tests.log)。

## A：阻抗模型与验证

M0 为 R+sL；M1 为 Rcu+sLinf+(Reddy||sLeddy)+(Rmot||sLmot||1/sCmot)。M2 增加一个正参数并联 RLC 串联支路。所有参数在 log 域拟合，soft_l1，多初值，DCR 固定五次读数均值，静置温度未知；假设系统不确定度 0.05 ohm，不能解释为仪器认证精度。宽扫和精扫各有相同总权重，组内按 log(f) 间隔积分权重，重复点先平均。

每 5 点留出第 3 点；训练/留出是交错频点验证，不能验证超出频带的外推。M2 只有两组留出均改善至少 15% 且 Jacobian 条件数 <1e6 才接受。最终选择 **{model}**。32 次（quick 模式 4 次）训练集 pairs bootstrap 表示拟合散布，不涵盖校准误差、温度漂移和模型结构误差；协方差/相关性是局部线性近似，见 [fit/fit.json](fit/fit.json)。

{table(['parameter','value'],[dict(parameter=n,value=f'{v:.9g}') for n,v in zip(names,p)])}

{table(['model','dataset','train_mean_relative','holdout_mean_relative','p95_relative','condition'],[
dict(model=m,dataset=k,train_mean_relative=f"{v['train_mean']:.6%}",holdout_mean_relative=f"{v['holdout_mean']:.6%}",
p95_relative=f"{v['quantiles']['p95']:.6%}",condition=f"{s['jacobian_condition']:.5g}")
for m,s in fitted['models'].items() for k,v in s['datasets'].items()])}

离散峰位（分别判断，不合并为一个共振频率）：

{table(['feature','measured_Hz','model_Hz','error_Hz'],[dict(feature=k,**v) for k,v in fitted['peaks'].items()])}

![宽频阻抗](plots/impedance_wide.png)
![精扫阻抗](plots/impedance_zoom.png)
![残差](plots/fit_residuals.png)

独立 ngspice AC 归一化复核 {ac['points']} 点，与解析函数最大差 {ac['max_error_ohm']:.5g} ohm。所有电阻/电感/电容严格正，串并联无源网络构造保证正实性；输入 AC 1 A 只是传递函数归一化。不能凭输入阻抗唯一辨识真实机械位移、辐射效率或 SPL。拟合网络的支路 f0={1/(2*np.pi*np.sqrt(p[5]*p[6])):.3f} Hz，不等于输入 |Z| 峰、谷或声压峰。

## B：电路与器件模型

正常路径：+5V → E(S8550)/S(CJ3401) → C/D → Rlimit → 蜂鸣器无源网络 → 1Ω → GND。二极管阴极接 C/D，阳极接 GND。关断路径：GND → 1N4148 阳极→阴极 → 负载顶端 → Rlimit → 线圈 → 1Ω → GND；OFF 电流仍经过采样电阻。正向线圈电流为采样电压/1Ω。

SPICE：PNP 行按 C B E；MOS 按 D G S B，PMOS 体端接 S。显式 PMOS 体二极管为 D 阳极→S 阴极。网表函数位于 src/uart_buzzer/spice.py，所有运行网表/日志在 spice/。每次运行创建新目录，旧结果和旧模型不覆盖。

**模型身份**：没有用户实际 S8550 厂家/封装；PNP_TYP 是人为构造的 Gummel-Poon 灵敏度模型（BF=150、TF=80ns、TR=300ns、结电容及电极电阻见网表），不是 onsemi SS8550 官方模型，更不代表用户实物。PMT 是按 CJ3401 静态低电流导通电阻量级构造的 level-1 模型，Cgs=877pF、Cgd=77pF、Cds=38pF，总 Ciss=954pF；常电容不能精确复现 Miller 非线性/Qg；其误差通过扫描探索，不能作为边沿最坏值保证。CJ3401 只作为待购候选，未假定用户持有。NMT 是典型 NMOS 对照，未冒充 Nexperia 2N7002HR 或 CJ 2N7002K 原厂模型。DF 为含 TT=4ns、Cjo=4pF 的典型二极管模型；反向恢复/正向恢复在本电流条件未实测。

5V 源内阻 0.25Ω、10µF+100nF，TX Thevenin 输出电阻默认 33Ω，扫描 20–150Ω；这不是 CH343 认证输出曲线，未建电流限幅。大于保守测试预算的瞬态电流只能标为待验证，不能把该波形当成 CH343 保证。外部 5V 存在且 VIO=0 时额外放入 TX→VIO ESD 钳位，演示危险路径；具体 ESD 模型仍未知。高阻态使用 1TΩ 断开主动输出；power_loss 案例还保留 VIO 下降时的钳位。

双 NMOS：TX→Q1 G，Q1 D 上拉 VIO→Q2 G，Q2 D 经 Rb→PNP B。TX 高为关断，TX 低为导通；高阻由 TX 上拉回 VIO，前提是 VIO 存在。VIO 掉电后 Q2 截止的典型结果不能代替真实寄生、栅极绝对额定和逆灌验证。历史单 NMOS G=VIO、S=TX、D 经 Rb→PNP B，其基极电流仍灌进 TX；5V 负载/3.3V VIO 时还存在阈值相关截止失败。

### 2kHz、50% 基准

{table(['case','Iavg_A','Irms_A','Ipeak_A','Pcu_W','Pswitch_W','Pdiode_W','TX_sink_peak_A'],[
{k:(f'{v:.6g}' if isinstance(v,float) else v) for k,v in r.items()} for r in rows if r['case'].endswith('_2k')])}

![PNP 2kHz](plots/pnp_2k.png)
![PMOS 2kHz](plots/pmos_2k.png)
![PNP UART](plots/pnp_bits.png)
![PMOS UART](plots/pmos_bits.png)

所有案例的启动峰值、稳态 RMS/均值、OFF RMS、二极管峰值/均值/反向恢复、电极/栅极电压、TX 电流、端口电流和支路损耗见 [all_cases.csv](tables/all_cases.csv)。RMS 使用非均匀 ngspice 时刻的梯形积分，端点插值；频谱再单独插值均匀采样。coil 与 shunt 的全周期一致性直接检查。被动回路能量核对包含电感/电容首末储能差，避免把启动储能当作损耗。开路/数值噪声下的相对能量误差不用于合格判定。

完整运行另以 Gear 积分复核921600 baud/PMOS/2kHz ΣΔ（[integration_method_check.json](integration_method_check.json)）。梯形积分的二极管电流含数值高频波纹，不能将它当作实物振铃/恢复电流实测：该代表点线圈RMS变化约0.00077%，二极管全记录RMS约38.1→34.7mA，峰值约90.5→86.7mA。主表保留原始梯形积分值与日志；线圈电流/铜耗在该复核中稳定，二极管动态指标仍须更多时间步长与实物校准。两种方法都不是器件最坏恢复保证。

### 故障与兼容性

{table(['case','Irms_A','Pcu_W','TX_sink_peak_A','Vcontrol_abs_peak_V'],[
{k:(f'{v:.6g}' if isinstance(v,float) else v) for k,v in r.items()} for r in rows if any(s in r['case'] for s in ('idle33','usb_dead','highZ','stuck_low'))])}

5V TTL 可直接反相控制两个首选拓扑；3.3V 高电平相对 5V 发射极/源极残留约 -1.7V，不能保证关断。外部负载电源独立、USB/VIO 失电时直接 PNP/PMOS 都有拉低控制端、持续导通和逆灌危险。单纯上拉不消除 ESD 通路；要求电源联动切断负载，或使用独立隔离且失电默认 OFF 的驱动/负载开关。软件停发不能作为该故障保护。

理想 DC 故障极限（忽略采样和开关压降）：5/40.80377={5/40.80377:.6g}A，铜耗={25/40.80377:.6g}W。考虑 1Ω 与限流时，以 37Ω、5.25V 冷态角点，33Ω：约 74mA、线圈 0.202W、限流电阻 0.180W；68Ω：约 49.5mA、线圈 0.091W、电阻 0.167W（解析近似）。实际 SPICE 角点见 sensitivity.csv。原厂未定义连续 DC 热许可，因此这些值仍未证明可无限时安全；实验选至少 0.5W 电阻且复核封装降额，独立硬件计时/断电保护不可省。

## C：UART 与声音相关电气指标

每帧起始 0、8 数据位 LSB first、停止 1；coil_on≈1−TX 只在开关跟随足够时成立。连续 0xFF 强制 10% ON，0x00 强制 90% ON；真正静音为 FIFO 排空后停发、TX 高。文件 notes_rests_sweep.bin 配套 JSON 调度表示真正停发的休止和 USB 空隙，单独发送 .bin 不会自动插入休止。

{pwl_note}

三个编码器：naive_byte_density 逐帧取整分配前置 ON 数据位；frame_aware_pdm 每帧 256 候选最小化帧内累计误差/帧末误差；frame_aware_sigma_delta 保留跨帧积分误差，逐帧对 256 字节作相同优化。均只输出可真实发送的字节，无二阶稳定性或全局最优声明。复杂度 O(256×10×frames)，耗时见 encoder_cost.csv。它是受约束一阶误差反馈搜索，不是理想任意 bitstream ΣΔ。

230400/460800/921600 baud 与 440/880/1760/2000/2150/2400/3520Hz 经过两种首选器件的真实 SPICE。短提示音、440→1760Hz 升两八度、音符切换、真正休止、1900–2400Hz 扫频与插空恢复也有字节/调度/波形。端到端目标是仿真线圈电流，离线编码搜索仍基于数字累计误差，未宣称完成全局电路反演最优。

基音与 2–5 次谐波用同时最小二乘正弦拟合，适用于非整周期；残差经 Hann FFT/ENBW=1.5 计算 100Hz–10kHz 带内噪声。THD 和残差比在电学域，不等同听觉失真。mechanical acoustic_proxy 为运动并联 RLC 支路电压的基音幅值，只是可配置假设的候选运动代理；无麦克风标定，所有绝对 SPL 列为 unknown。

完整编码/电流指标见 [uart_comparison.csv](tables/uart_comparison.csv)，脉冲缺损/延迟见 all_cases.csv。边沿判据为输出节点越过 2.5V，相对命令边沿的最大延迟和未到达命令状态的脉冲计数；不是实测传播延迟。高波特率是否优于低波特率由这些模拟指标共同判断，实际驱动可设波特率、USB FIFO/间隙与实时吞吐仍需验证。

{findings}

本次结果没有显示复杂编码器在所有指标上同时获胜。例如921600 baud、2kHz、PMOS/Rg=220Ω、偏置0.5/幅度0.35：naive 的基音峰值约23.73mA、THD(2–5)约3.66%、铜耗132mW；一阶帧感知ΣΔ约25.18mA、8.95%、167mW，而带内残差/基音比由约2.35%下降到1.46%。数字累计误差更小不保证经过开关后谐波更低；这是保留电路联合评估的原因。

Rg=220Ω、Rgs=10kΩ 的已测模型最大延迟约0.434µs、TX瞬时灌流约19.4mA；Rg=470Ω 降到约9.84mA，却延迟约0.800µs，已占1.085µs位宽很大比例。这不是CH343已保证的瞬态电流，两个候选都须实测，必要时加驱动器。少部件不等于已经满足TX预算。

与低偏置及限流联合后，PMOS、921600 baud、ΣΔ、Rlimit=33Ω、偏置0.3/幅度0.2 的模型结果约22.5mA RMS、20.7mW铜耗、13.6mA基音峰值、1.31% THD；以较低声音代理换取铜耗。这些指标仅适用于该具体网格点，未完成声学/热认证，不能与未限流点作相同声压的等效比较。

## D：联合扫描与条件性建议

Rg={{0,47,100,220,470}}Ω、Rgs={{1k,4.7k,10k,22k}}Ω；Rb={{330,470,680,1000,2200}}Ω、Rpull={{1k,2.2k,4.7k,10k}}Ω；双 NMOS 上拉 470/1000/2200Ω、TX 上拉10k/22k/47kΩ；Rlimit={{0,10,22,33,47,68}}Ω，二极管串联钳位电阻 0/22Ω；偏置 0.3/0.5、幅度 0.15/0.2/0.35（密度截断到0.1–0.9）和波特率，另有 PNP 防饱和 Schottky 对照。这里只扫描子网格，未穷举全部组合或元件成本/库存。

72 组随机敏感度（quick 9 组）分三拓扑，包含 Rcu=37/40.80377/47Ω、5V±5%、β/TF/TR、PMOS Vth/Ciss/KP、TX Rout、NMOS Vth、电感±20% 和二极管 IS，再加54个 DC 确定角点。随机扫描的最大值是“已采样最坏”，不是所有联合容差的严格极限；超出测量频带的模型结构不确定度未被 Monte Carlo 消除。统计见 monte_carlo_summary.json、逐行参数见 sensitivity.csv。

代表性921600 baud边沿扫描（PMOS Rgs=10kΩ、PNP Rpull=4.7kΩ）：

{edge_scan}

{worst_text}

![Pareto](plots/pareto.png)

完整器件端口及控制/上拉/输出电阻/供电电阻损耗见 [additional_losses.csv](tables/additional_losses.csv)，直接驱动提供全源端功率审计。热参数条件性扫描见 [thermal_assumptions.csv](tables/thermal_assumptions.csv)；该图固定假设铜耗，不含已标定热反馈，不能当作实测温升。

![假设热模型](plots/thermal_assumptions.png)

Pareto 比较 mechanical proxy、铜耗、TX 平均绝对电流、器件损耗、电压峰值和延迟；每行都保留硬约束 **unverified**。没有以未定义的 50mA RMS、0.2W 铜耗或绝对声压门槛伪造通过。拓扑数量/采购价格未确认，不参与数值 Pareto。

CSV 中 `TX_assumed_5mA_peak_budget_pass` 使用人为的5mA峰值筛选预算，不是WCH官方额定。在此假设下，直接PNP的Rb=680Ω/330Ω及本次PMOS栅极扫描均不能通过；它们保留作实验候选，不能推荐为已合格的长期直接驱动。预算确认后须重新判定。双NMOS虽消除了PNP基极直流负担，其TX瞬态也需检查；不能只凭静态电流较小宣布通过。

**建议实验起点而非最终定值**：优先 CJ3401 候选，Rg=220Ω、Rgs=10kΩ，100nF+10µF，1Ω 测量，33Ω 或68Ω/≥0.5W 串联实验保护，供电联动和独立时限保护。Rg 的选择需同时核对 TX 峰值与 1.085µs 位内有效 ON/OFF，470Ω 可能更温和但关断边沿需比较。不能把 Rg=0 的理想源峰值当成芯片可承受。若实际 TX 无法驱动大栅容，可改有明确电流能力和掉电行为的驱动器。

直接 S8550 以 Rb=680Ω、Rpull=4.7kΩ 作对照，其约6mA 基极灌流不是已测允许值。Rb=330Ω 的约13mA 与关断储存尾巴都需核对。未知 CH343 模块预算下无法认证直接 PNP；不要仅因少一个元件选择它。双 NMOS 缓冲消除 TX 的 PNP 基极直流负担，但增加反相节点延迟、静态耗电和器件身份验证需求。历史单 NMOS 不推荐作为隔离缓冲。

**二极管界限**：所引用 Nexperia 1N4148 DO-35 文档给 100V、重复峰值450mA、25℃/规定引线安装下连续200mA与500mW；不代表所有 1N4148 封装/厂商通用。按各行 diode_peak_A/diode_avg_A/Pdiode_W/diode_reverse_V 逐项比较；额定、温度降额与PCB散热均须按实物。串联22Ω钳位加速衰减，同时增大开关 |VCE/VDS|，不得只看尾音改善。CJ3401 文档 VDS=-30V、VGS=±12V；S8550 绝对额定因身份缺失不可最终打勾。

## 下一轮实测与保护

1. 不接蜂鸣器先测 TX idle、TX LOW、VIO 和5V；确认共地。用已知安全负载阶梯测 TX VOL/IOL、VOH/IOH，确认模块不是只切电源针。记录 CH343 型号/版本、模块板图、真实 S8550 品牌/封装/引脚和 CJ3401 是否已购。
2. 外部5V高边接线如前述。示波器所有普通地夹只接同一 GND；不能将地夹接高边 E/S、C/D、B/G 形成短路。Vshunt 对 GND测量完整电流，其他浮动差分电压用差分探头或合规双通道相减。
3. 先33/68Ω、限时2kHz50%，采样 TX、B/C或G/D、Vshunt、二极管；记录启动峰值、稳态 Iavg/Irms、开关压降、二极管尖峰。确认关断和独立硬件故障时限，再比较460800/921600真实8N1，检查 FIFO 排空、字节间隙和USB拔出。
4. 独立外部5V下，先用假负载验证USB/VIO失电与TX高阻，限流观察反向灌电。没有证据证明关断前不接蜂鸣器长期测试。
5. 麦克风距10cm保持固定几何，记录2kHz基准、各音符相对声压与听感；补录冷/热态DCR、环境/表面温度及时间。铜温系数0.0039/K仅用于DCR温度推断，热阻热容未标定前不报告已验证温升。热模型灵敏度输出均为假设条件，不能作为安全门槛。

## 原厂参考与访问限制

- [CJ3401 Rev2.1](https://www.jscj-elec.com/gallery/file/CJ3401%20SOT-23%20V2.1.pdf)：G/S/D=1/2/3、RDS(on)≤75mΩ@−4.5V、Ciss约954pF。并未取得厂家完整动态 SPICE 模型。
- [Nexperia 1N4148/1N4448](https://assets.nexperia.com/documents/data-sheet/1N4148_1N4448.pdf)：上述DO-35限定值；本项目实际物料待确认。
- [Nexperia 2N7002H](https://assets.nexperia.com/documents/data-sheet/2N7002H.pdf)：对照身份，NMT未冒充原厂。
- [onsemi SS8550](https://www.onsemi.com/pdf/datasheet/ss8550-d.pdf)：本次访问403，未取得可验证原厂模型；不代表实物S8550。
- [WCH CH343官方手册入口](https://www.wch-ic.com/downloads/CH343DS1_PDF.html)：官方索引可确认支持多电平和标称速率；文档内容因JS入口未在本次读取验证。其他镜像搜索虽出现15mA/VOL值，未将未核对版本的镜像值当作官方芯片/模块保证。本研究采用参数化源阻抗与待测预算。

## 复现与局限

项目根目录 `python run.py`；用 `--dll PATH` 指定兼容 ngspice 动态库，或安装CLI；`python scripts/run_tests.py` 运行测试。requirements.txt 固定Python包版本。每次新建结果目录，environment.json记录运行参数/版本；`.cir/.log`是原始仿真证据，`.npz`是保留案例的全部波形，CSV是积分/谱/优化结果。

此研究不能代替5V大信号阻抗/磁饱和测试，不能保证绝对SPL、长时温升和USB流无插空。官方动态器件模型、实物引脚/批次和TX限流行为均不充分；因此本次交付是有复现证据的条件性评估，不是可无防护长期播放的定型认证。
'''
    (out/'report.md').write_text(text,encoding='utf-8')
    (root/'report.md').write_text('# 最新研究报告\n\n完整报告、证据和图表见 ['+str(out.relative_to(root).as_posix())+'/report.md]('+str(out.relative_to(root).as_posix())+'/report.md)。\n\n本项目从测量拟合到晶体管/二极管ngspice与真实8N1联合评估；硬件最终认证、绝对SPL和温升仍未验证。\n',encoding='utf-8')
