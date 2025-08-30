import io, sys
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import openpyxl
import os

# --- Constants and Configuration ---
KMH_TO_MS = 1 / 3.6
G = 9.81
DEFAULT_RHO = 1.2
TIME_STEP = 1

# Optional streamlit
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# -----------------------
# Drive cycle utilities
# -----------------------
def default_hwfet_cycle(duration_s=600, dt=1.0):
    t = np.arange(0, duration_s + dt, dt)
    speed = 20 + 30 * np.sin(2 * np.pi * t / 180) + 8 * np.sin(2 * np.pi * t / 40) + 5 * np.sin(2 * np.pi * t / 10)
    speed = np.clip(speed, 0, 120)
    mid = len(t) // 2
    speed[mid:mid + 5] = np.minimum(speed[mid:mid + 5], 0.5)
    return pd.DataFrame({'time_s': t, 'speed_kmh': speed})

def load_cycle_from_bytes(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a .csv or .xlsx file.")

    cols = [c.lower() for c in df.columns]

    if 'time_s' in cols and 'speed_kmh' in cols:
        df = df.rename(columns={df.columns[cols.index('time_s')]: 'time_s', df.columns[cols.index('speed_kmh')]: 'speed_kmh'})
    elif 'time' in cols and 'speed' in cols:
        df = df.rename(columns={df.columns[cols.index('time')]: 'time_s', df.columns[cols.index('speed')]: 'speed_kmh'})
    else:
        df = df.iloc[:, :2].copy()
        df.columns = ['time_s', 'speed_kmh']

    return df.sort_values('time_s').reset_index(drop=True)

def extend_drive_cycle(df, target_duration=3600):
    original_duration = df['time_s'].max() + TIME_STEP
    if original_duration >= target_duration:
        return df

    num_repeats = math.ceil(target_duration / original_duration)
    extended_dfs = [df.copy() for _ in range(num_repeats)]

    current_time_offset = 0
    for i in range(1, num_repeats):
        current_time_offset += original_duration
        extended_dfs[i]['time_s'] += current_time_offset

    extended_df = pd.concat(extended_dfs, ignore_index=True)
    extended_df = extended_df[extended_df['time_s'] < target_duration]

    return extended_df

# -----------------------
# Simple motor map
# -----------------------
def simple_motor_map(max_power_w=20000, max_torque_nm=200, knee_rpm=3000):
    omega_knee = knee_rpm * 2 * math.pi / 60.0
    def torque_limit(omega):
        omega = np.asarray(omega, dtype=float)
        tau = np.where(omega <= omega_knee, max_torque_nm,
                       np.maximum(0.0, max_power_w / (omega + 1e-9)))
        return tau
    def eff_lookup(torque, omega):
        torque = np.abs(np.asarray(torque, dtype=float))
        omega = np.asarray(omega, dtype=float)
        tnorm = np.clip(torque / (max_torque_nm + 1e-9), 0, 2)
        onorm = np.clip(omega / (omega_knee + 1e-9), 0, 3)
        eff = 0.6 + 0.35 * np.exp(-((tnorm - 0.4) ** 2) / 0.2) * np.exp(-((onorm - 0.8) ** 2) / 0.7)
        return np.clip(eff, 0.45, 0.95)
    return torque_limit, eff_lookup

# -----------------------
# Core simulator
# -----------------------
def run_simulation(df_cycle,
                   mass_kg,
                   Cd, Cr, Af,
                   wheel_radius_m,
                   battery_wh,
                   soc0=0.9,
                   battery_max_charge_w=5000.0,
                   battery_max_discharge_w=60000.0,
                   regen_power_cap_w=3000.0,
                   motor_max_power_w=40000.0,
                   motor_max_torque_nm=250.0,
                   motor_knee_rpm=3000.0,
                   ambient_rho=DEFAULT_RHO,
                   slope_deg=0.0,
                   gear_ratio=1.0,
                   use_motor_map=True):
    df = df_cycle.copy().reset_index(drop=True)
    dt = df['time_s'].diff().fillna(df['time_s'].iloc[1] - df['time_s'].iloc[0]).values
    v_ms = df['speed_kmh'].values * KMH_TO_MS
    dv = np.concatenate(([0.0], np.diff(v_ms)))
    a = dv / dt

    torque_limit_fn, eff_fn = simple_motor_map(motor_max_power_w, motor_max_torque_nm, motor_knee_rpm)

    # Forces
    F_aero = 0.5 * ambient_rho * Cd * Af * v_ms ** 2
    F_roll = mass_kg * G * Cr * np.sign(v_ms + 1e-9)
    theta = math.radians(slope_deg)
    F_slope = mass_kg * G * math.sin(theta) * np.ones_like(v_ms)
    F_iner = mass_kg * a

    # required tractive force
    F_trac_req = F_aero + F_roll + F_slope + F_iner

    # wheel & motor torques and speeds
    torque_wheel_req = F_trac_req * wheel_radius_m
    torque_motor_req = torque_wheel_req / gear_ratio
    omega_wheel = np.where(v_ms > 0, v_ms / wheel_radius_m, 0.0)
    omega_motor = omega_wheel * gear_ratio

    # apply torque limits (motor)
    tau_lim = torque_limit_fn(omega_motor)
    torque_motor_capped = np.clip(torque_motor_req, -tau_lim, tau_lim)
    torque_wheel_capped = torque_motor_capped * gear_ratio
    F_trac_capped = torque_wheel_capped / wheel_radius_m

    # mechanical and electrical powers
    P_wheel = F_trac_capped * v_ms
    effs = eff_fn(torque_motor_capped, omega_motor) if use_motor_map else np.full_like(P_wheel, 0.9)
    P_elec = np.zeros_like(P_wheel)

    for i, Pw in enumerate(P_wheel):
        if Pw >= 0:
            Pe = Pw / max(effs[i], 1e-6)
            Pe = min(Pe, motor_max_power_w, battery_max_discharge_w)
            P_elec[i] = Pe
        else:
            recovered = -Pw * effs[i]
            recovered = min(recovered, regen_power_cap_w, battery_max_charge_w)
            P_elec[i] = -recovered

    # energy integration (Wh)
    energy_flow_wh = np.cumsum(P_elec * dt) / 3600.0
    soc = soc0 - energy_flow_wh / battery_wh
    soc = np.clip(soc, 0.0, 1.0)

    # assemble results
    res = df.copy()
    res['dt_s'] = dt
    res['v_ms'] = v_ms
    res['v_kmh'] = df['speed_kmh']
    res['acc_m_s2'] = a
    res['F_aero_N'] = F_aero
    res['F_roll_N'] = F_roll
    res['F_slope_N'] = F_slope
    res['F_iner_N'] = F_iner
    res['F_trac_req_N'] = F_trac_req
    res['F_trac_capped_N'] = F_trac_capped
    res['torque_wheel_req_Nm'] = torque_wheel_req
    res['torque_motor_req_Nm'] = torque_motor_req
    res['torque_motor_capped_Nm'] = torque_motor_capped
    res['omega_motor_rad_s'] = omega_motor
    res['P_wheel_W'] = P_wheel
    res['P_elec_W'] = P_elec
    res['motor_eff'] = effs
    res['energy_flow_wh'] = energy_flow_wh
    res['soc'] = soc

    # summary
    distance_km = float(np.sum(v_ms * dt) / 1000.0)
    total_energy_wh = float(energy_flow_wh[-1])
    wh_per_km = float(total_energy_wh / distance_km) if distance_km > 0 else np.nan
    regen_recovered_wh = -float(np.sum(res.loc[res['P_elec_W'] < 0, 'P_elec_W'] * res.loc[res['P_elec_W'] < 0, 'dt_s']) / 3600.0)
    peak_torque_Nm = float(np.nanmax(np.abs(res['torque_motor_capped_Nm'])))
    peak_wheel_power_w = float(np.nanmax(res['P_wheel_W']))
    peak_elec_w = float(np.nanmax(res['P_elec_W'])) if np.any(res['P_elec_W'] >= 0) else 0.0

    summary = {
        'distance_km': distance_km,
        'total_energy_wh': total_energy_wh,
        'wh_per_km': wh_per_km,
        'battery_wh': battery_wh,
        'regen_recovered_wh': regen_recovered_wh,
        'peak_torque_Nm': peak_torque_Nm,
        'peak_wheel_power_W': peak_wheel_power_w,
        'peak_elec_W': peak_elec_w,
        'soc_start': float(res['soc'].iloc[0]),
        'soc_end': float(res['soc'].iloc[-1])
    }
    return res, summary

# -----------------------
# Slope steady performance
# -----------------------
def slope_performance_table(mass_kg, Cd, Cr, Af, rho, wheel_radius_m, gear_ratio, motor_eff, regen_eff, battery_wh, speeds_kmh, slopes_deg):
    rows = []
    for s in slopes_deg:
        for sp in speeds_kmh:
            v_ms = sp * KMH_TO_MS
            F_aero = 0.5 * rho * Cd * Af * v_ms * v_ms
            F_roll = mass_kg * G * Cr
            F_slope = mass_kg * G * math.sin(math.radians(s))
            F_trac = F_aero + F_roll + F_slope
            P_wheel = F_trac * v_ms
            torque_wheel = F_trac * wheel_radius_m
            torque_motor = torque_wheel / gear_ratio
            P_elec = P_wheel / motor_eff if P_wheel >= 0 else P_wheel * regen_eff
            time_1km = 1000.0 / v_ms if v_ms > 0 else np.nan
            wh_per_km = P_elec * time_1km / 3600.0 if v_ms > 0 else np.nan
            est_range_km = battery_wh / wh_per_km if (wh_per_km and wh_per_km > 0) else np.nan
            rows.append({
                'slope_deg': s,
                'speed_kmh': sp,
                'F_trac_N': F_trac,
                'torque_wheel_Nm': torque_wheel,
                'torque_motor_Nm': torque_motor,
                'P_wheel_W': P_wheel,
                'P_elec_W': P_elec,
                'Wh_per_km': wh_per_km,
                'est_range_km': est_range_km
            })
    return pd.DataFrame(rows)

# -----------------------
# Powertrain recommendation logic
# -----------------------
def recommend_powertrain(summary, desired_range_km=None, margin=1.2):
    peak_elec = summary['peak_elec_W']
    peak_torque = summary['peak_torque_Nm']
    whpkm = summary['wh_per_km'] if summary['wh_per_km'] and not np.isnan(summary['wh_per_km']) else None

    suggested_motor_peak_w = max(150, peak_elec * margin)
    suggested_motor_cont_w = max(100, suggested_motor_peak_w * 0.35)

    if desired_range_km and whpkm:
        suggested_battery_wh = whpkm * desired_range_km
    else:
        suggested_battery_wh = summary.get('battery_wh', None)

    recommendations = {
        'suggested_motor_peak_w': suggested_motor_peak_w,
        'suggested_motor_cont_w': suggested_motor_cont_w,
        'suggested_motor_torque_nm': peak_torque * margin,
        'suggested_battery_wh': suggested_battery_wh
    }

    tiers = []
    tiers.append({'name': 'Light', 'motor_peak_w': suggested_motor_peak_w * 0.6, 'battery_wh': suggested_battery_wh * 0.6 if suggested_battery_wh else None})
    tiers.append({'name': 'Mid', 'motor_peak_w': suggested_motor_peak_w, 'battery_wh': suggested_battery_wh})
    tiers.append({'name': 'Heavy', 'motor_peak_w': suggested_motor_peak_w * 1.5, 'battery_wh': suggested_battery_wh * 1.5 if suggested_battery_wh else None})
    return recommendations, tiers

# -----------------------
# Plot helpers & insights
# -----------------------
def plot_speed_time(res_df):
    return px.line(res_df, x='time_s', y='v_kmh', title='1. Speed vs Time', labels={'time_s': 'Time (s)', 'v_kmh': 'Speed (km/h)'})

def plot_tractive_force(res_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_df['time_s'], y=res_df['F_trac_req_N'], name='Requested Tractive Force (N)'))
    fig.add_trace(go.Scatter(x=res_df['time_s'], y=res_df['F_trac_capped_N'], name='Capped (motor limit) (N)'))
    fig.update_layout(title='2. Tractive Force vs Time', xaxis_title='Time (s)', yaxis_title='Force (N)')
    return fig

def plot_power_time(res_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_df['time_s'], y=res_df['P_wheel_W'], name='Wheel Power (W)'))
    fig.add_trace(go.Scatter(x=res_df['time_s'], y=res_df['P_elec_W'], name='Electrical Power (W)'))
    fig.update_layout(title='3. Power vs Time', xaxis_title='Time (s)', yaxis_title='Power (W)')
    return fig

def plot_soc_time(res_df):
    return px.line(res_df, x='time_s', y='soc', title='4. Battery SOC vs Time', labels={'time_s': 'Time (s)', 'soc': 'SOC'})

def plot_force_breakdown(res_df):
    fig = px.line(res_df, x='time_s', y=['F_aero_N', 'F_roll_N', 'F_slope_N', 'F_iner_N'],
                  title='5. Force Breakdown',
                  labels={'value': 'Force (N)', 'variable': 'Force Type'})
    fig.update_traces(selector=dict(name='F_aero_N'), name='Aerodynamic')
    fig.update_traces(selector=dict(name='F_roll_N'), name='Rolling Resistance')
    fig.update_traces(selector=dict(name='F_slope_N'), name='Slope')
    fig.update_traces(selector=dict(name='F_iner_N'), name='Inertial')
    return fig

def plot_energy_consumption(res_df):
    return px.line(res_df, x='time_s', y='energy_flow_wh', title='6. Cumulative Energy Consumption', labels={'time_s': 'Time (s)', 'energy_flow_wh': 'Energy (Wh)'})

def plot_motor_torque_speed(res_df):
    return px.line(res_df, x='time_s', y='torque_motor_req_Nm', title='7. Motor Torque vs Time', labels={'time_s': 'Time (s)', 'v_kmh': 'Motor Torque'})

def plot_power_demand_hist(res_df):
    fig = px.histogram(res_df, x='P_elec_W', nbins=50, title='8. Electrical Power Demand Histogram')
    return fig

def plot_accel_vs_speed(res_df):
    return px.line(res_df, x='time_s', y='acc_m_s2', title='9. Acceleration vs Time', labels={'v_kmh': 'Speed (km/h)', 'acc_m_s2': 'Acceleration (m/s²)'})

def plot_regen_power_time(res_df):
    regen_df = res_df[res_df['P_elec_W'] < 0]
    return px.line(regen_df, x='time_s', y='P_elec_W', title='10. Regenerative Power', labels={'time_s': 'Time (s)', 'P_elec_W': 'Power (W)'})

def insight_block_for_plot(plot_name, res_df=None, summary=None):
    obs = []; rec = []; concl = []
    if plot_name == '1. Speed vs Time':
        obs.append(f"Peak speed: **{res_df['v_kmh'].max():.1f} km/h**; mean: **{res_df['v_kmh'].mean():.1f} km/h**.")
        rec.append("To increase range, reduce sustained high speeds. Aerodynamic improvements are most effective here.")
        concl.append("The speed profile is the primary driver of energy consumption. High-speed segments require significantly more power due to aerodynamic drag.")
    elif plot_name == '2. Tractive Force vs Time':
        obs.append(f"Peak requested force: **{float(np.max(np.abs(res_df['F_trac_req_N']))):.0f} N**.")
        rec.append("Soften accelerations and reduce vehicle mass to lower peak force demands.")
        concl.append("High force peaks require a high-torque motor or a more aggressive gear ratio. This is critical for hill climbing and rapid acceleration.")
    elif plot_name == '3. Power vs Time':
        obs.append(f"Peak wheel power: **{summary['peak_wheel_power_W']:.0f} W**; Peak electrical draw: **{summary['peak_elec_W']:.0f} W**.")
        rec.append("Select a motor and battery that can reliably handle the peak electrical power demand with a safety margin.")
        concl.append("The average electrical power determines your total energy consumption, while peak power dictates the required size of your motor and battery controller.")
    elif plot_name == '4. Battery SOC vs Time':
        obs.append(f"SOC starts at **{summary['soc_start']:.2f}** and ends at **{summary['soc_end']:.2f}**; recovered energy: **{summary['regen_recovered_wh']:.1f} Wh**.")
        rec.append("If the final SOC is too low, increase battery capacity or reduce the average power consumption.")
        concl.append("This plot shows the battery's state of health over the cycle. A steep drop indicates high power demand and low range. The recovered energy highlights the effectiveness of regenerative braking.")
    elif plot_name == '5. Force Breakdown':
        obs.append("Aerodynamic drag and rolling resistance dominate forces at steady speeds; inertial force spikes during acceleration.")
        rec.append("To improve efficiency, focus on reducing mass for city driving and lowering Cd/Af for high-speed use.")
        concl.append("Understanding the force breakdown helps you target specific design improvements for your use case.")
    elif plot_name == '6. Cumulative Energy Consumption':
        obs.append(f"Total energy used over the cycle: **{summary['total_energy_wh']:.1f} Wh**.")
        rec.append("Analyze where the slope is steepest to find segments of the cycle with high energy use (e.g., hard acceleration, high speed).")
        concl.append("This plot provides a clear picture of total energy consumed, which directly relates to your vehicle's range.")
    elif plot_name == '7. Motor Torque vs Time':
        obs.append("The plot shows the relationship between required torque and motor speed throughout the cycle. The 'capped' line represents the motor's physical limit.")
        rec.append("If the 'required' curve often exceeds the 'capped' curve, your motor is undersized for the task. Consider a more powerful motor.")
        concl.append("This graph is a motor's 'operating map' and is crucial for selecting a motor that can handle the full range of operational demands.")
    elif plot_name == '8. Electrical Power Demand Histogram':
        obs.append("Most of the time, the power demand is low, but there are frequent high-power spikes.")
        rec.append("Ensure your motor controller and battery can handle the most frequent power demands, as well as the infrequent peaks.")
        concl.append("This histogram gives a statistical overview of your power usage, helping you choose a component that is efficient for your average use and robust for peak loads.")
    elif plot_name == '9. Acceleration vs Time':
        obs.append("The scatter plot shows periods of positive acceleration (speeding up) and negative acceleration (slowing down).")
        rec.append("A good trick scooter needs strong positive acceleration at low to mid-range speeds. Check that the required acceleration is met by your motor.")
        concl.append("This plot directly visualizes the vehicle's dynamic performance, which is key for a trick scooter.")
    elif plot_name == '10. Regenerative Power':
        obs.append(f"The recovered power during braking peaks at approximately **{-res_df['P_elec_W'].min():.0f} W**.")
        rec.append("Maximize regen power by choosing a motor/controller that is efficient in regen mode and a battery that can accept high charge rates.")
        concl.append("Regen is most effective in stop-and-go conditions. It significantly improves efficiency and reduces brake wear.")

    return dict(observations=obs, recommendations=rec, conclusions=concl)

# -----------------------
# Streamlit App UI
# -----------------------
def run_streamlit_app():
    st.set_page_config(layout='wide', page_title='E-Scooter & E-Bike Powertrain Designer')
    st.title("🛴 E-Scooter Powertrain Designer")
    st.write("Simulate performance and select optimal components based on your drive cycles.")

    if 'results' not in st.session_state:
        st.session_state.results = {}

    with st.sidebar:
        st.header("Vehicle Parameters")
        st.markdown("_Input your vehicle's physical properties._")
        defaults = {
            'Electric scooter/trike': {'mass': 120, 'Cd': 0.9, 'Cr': 0.008, 'Af': 0.4, 'wheel_r': 0.18, 'battery_wh': 2000, 'motor_w': 3000},
            'Electric bicycle': {'mass': 100, 'Cd': 1.0, 'Cr': 0.005, 'Af': 0.5, 'wheel_r': 0.34, 'battery_wh': 720, 'motor_w': 500},
            'Electric motorcycle': {'mass': 250, 'Cd': 0.6, 'Cr': 0.012, 'Af': 0.6, 'wheel_r': 0.32, 'battery_wh': 12000, 'motor_w': 20000},
            'Electric car': {'mass': 1500, 'Cd': 0.28, 'Cr': 0.01, 'Af': 2.2, 'wheel_r': 0.30, 'battery_wh': 60000, 'motor_w': 60000}
        }
        ev_type = st.selectbox("Select EV Type for default parameters", list(defaults.keys()))
        d = defaults[ev_type]

        mass = st.number_input("Vehicle mass (kg)", value=float(d['mass']), min_value=20.0, help="Vehicle + Rider mass")
        Cd = st.number_input("Drag coefficient (Cd)", value=float(d['Cd']), format="%.3f")
        Cr = st.number_input("Rolling resistance (Cr)", value=float(d['Cr']), format="%.4f")
        Af = st.number_input("Frontal area ($m^2$)", value=float(d['Af']), format="%.3f")
        wheel_r = st.number_input("Wheel radius (m)", value=float(d['wheel_r']), format="%.3f")
        gear_ratio = st.number_input("Gear ratio (motor:wheel)", value=1.0, format="%.3f")
        rho_air = st.number_input("Air density (kg/m^3)", value=DEFAULT_RHO)
        slope_deg = st.number_input("Constant slope during cycle (deg)", value=0.0, min_value=-15.0, max_value=15.0, step=0.5)

        st.markdown("---")
        st.header("Motor & Battery Specs")
        st.markdown("_Adjust these values to see their impact._")
        battery_wh = st.number_input("Battery capacity (Wh) - 0 to auto-estimate", value=float(d['battery_wh']), min_value=0.0, step=100.0)
        soc0 = st.slider("Initial SOC (fraction)", 0.0, 1.0, 0.9, 0.01)
        motor_max_power_w = st.number_input("Motor peak power (W)", value=float(d['motor_w']))
        motor_max_torque_nm = st.number_input("Motor torque @ zero speed (Nm)", value=20.0, help="Important for acceleration")
        motor_knee_rpm = st.number_input("Motor knee RPM", value=3000.0, help="Speed at which torque begins to drop")
        regen_cap_w = st.number_input("Max regen power cap (W)", value=3000.0)

        st.markdown("---")
        st.header("Drive Cycles")
        st.markdown("_Upload multiple cycles to compare performance._")
        uploaded_files = st.file_uploader("Upload CSV or XLSX files", type=['csv', 'xlsx'], accept_multiple_files=True)
        use_default = st.checkbox("Use built-in HWFET-like cycle", value=True)

        run_sim_button = st.button("Run / Re-run simulation")

    if run_sim_button:
        st.session_state.results = {}
        df_cycles = {}
        if use_default:
            df_cycles['Default Cycle'] = default_hwfet_cycle()

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                try:
                    df_cycles[file_name] = load_cycle_from_bytes(uploaded_file)
                except Exception as e:
                    st.error(f"Error processing file '{file_name}': {e}")

        if not df_cycles:
            st.warning("Please select a default cycle or upload a file to run the simulation.")
        else:
            for name, df_cycle in df_cycles.items():
                if battery_wh <= 0:
                    # quick sim to get Wh/km
                    temp_batt = 1000.0
                    qres, qsum = run_simulation(df_cycle, mass, Cd, Cr, Af, wheel_r, temp_batt, soc0, regen_cap_w, motor_max_power_w, regen_cap_w, motor_max_power_w, motor_max_torque_nm, motor_knee_rpm, rho_air, slope_deg, gear_ratio)
                    whpkm = qsum['wh_per_km'] if qsum['wh_per_km'] and not np.isnan(qsum['wh_per_km']) else 100.0
                    target_range = 150 if ev_type == 'Electric car' else (120 if ev_type == 'Electric motorcycle' else (60 if ev_type == 'Electric bicycle' else 50))
                    auto_battery_wh = max(200.0, whpkm * target_range)
                else:
                    auto_battery_wh = battery_wh

                res_df, summary = run_simulation(
                    df_cycle, mass, Cd, Cr, Af, wheel_r, auto_battery_wh, soc0, regen_cap_w, motor_max_power_w,
                    regen_cap_w, motor_max_power_w, motor_max_torque_nm, motor_knee_rpm, rho_air, slope_deg, gear_ratio
                )
                st.session_state.results[name] = (res_df, summary)
            st.success("All selected cycles have been simulated.")

    st.header("Simulation Results")

    if st.session_state.results:
        cycle_names = list(st.session_state.results.keys())
        selected_cycle = st.selectbox("Select a drive cycle to view:", cycle_names)

        res_df, summary = st.session_state.results[selected_cycle]

        st.subheader(f"Summary for **{selected_cycle}**")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("Distance", f"{summary['distance_km']:.1f} km")
        with col2: st.metric("Energy Consumption", f"{summary['total_energy_wh']:.1f} Wh")
        with col3: st.metric("Avg. Wh/km", f"{summary['wh_per_km']:.1f} Wh/km")
        with col4: st.metric("Peak Electrical Power", f"{summary['peak_elec_W']:.0f} W")
        with col5: st.metric("Est. Range", f"{summary['battery_wh']/summary['wh_per_km']:.0f} km")

        st.markdown("---")

        # Displaying plots and insights
        tab1, tab2, tab3 = st.tabs(["Performance Graphs", "Detailed Calculations", "Powertrain Selector"])

        with tab1:
            st.subheader("Performance Graphs")
            graphs_per_row = 2

            graph_funcs = [
                plot_speed_time, plot_tractive_force, plot_power_time, plot_soc_time,
                plot_force_breakdown, plot_energy_consumption, plot_motor_torque_speed,
                plot_power_demand_hist, plot_accel_vs_speed, plot_regen_power_time
            ]

            for i in range(len(graph_funcs)):
                if i % graphs_per_row == 0:
                    cols = st.columns(graphs_per_row)

                with cols[i % graphs_per_row]:
                    fig = graph_funcs[i](res_df)
                    st.plotly_chart(fig, use_container_width=True)
                    plot_name = fig.layout.title.text
                    insp = insight_block_for_plot(plot_name, res_df, summary)
                    st.markdown(f"**Observations**")
                    for o in insp['observations']: st.write("-", o)
                    st.markdown(f"**Recommendations**")
                    for r in insp['recommendations']: st.write("-", r)
                    st.markdown(f"**Conclusion**")
                    for c in insp['conclusions']: st.write("-", c)

        with tab2:
            st.subheader("Summary Metrics")
            st.write(f"- **Trip distance:** {summary['distance_km']:.2f} km")
            st.write(f"- **Net energy used:** {summary['total_energy_wh']:.1f} Wh")
            st.write(f"- **Average consumption:** {summary['wh_per_km']:.1f} Wh/km")
            st.write(f"- **Estimated range on this cycle:** {summary['battery_wh']/summary['wh_per_km']:.1f} km")
            st.write(f"- **Peak motor torque:** {summary['peak_torque_Nm']:.1f} Nm")
            st.write(f"- **Peak electrical draw:** {summary['peak_elec_W']:.0f} W")
            st.write(f"- **Regenerative energy recovered:** {summary['regen_recovered_wh']:.1f} Wh")

            st.markdown("---")
            st.subheader("Vehicle Performance on Slopes")
            st.write("This table shows the required motor power and torque to maintain a constant speed on various slopes.")
            cruise_speed = st.number_input("Cruise speed for slope table (km/h)", value=30.0)
            slope_df = slope_performance_table(mass, Cd, Cr, Af, rho_air, wheel_r, gear_ratio, 0.9, 0.6, battery_wh, [cruise_speed], list(range(0, 16)))
            st.dataframe(slope_df.style.format({'F_trac_N':'{:.1f}','torque_wheel_Nm':'{:.1f}','P_wheel_W':'{:.0f}','P_elec_W':'{:.0f}','Wh_per_km':'{:.2f}','est_range_km':'{:.0f}'}))

            st.markdown("---")
            st.subheader("How Regenerative Braking Works")
            st.write("During braking (negative wheel power), the motor acts as a generator, converting kinetic energy back into electrical energy. This energy is then stored in the battery, reducing overall energy consumption. This app models regen as being limited by the motor's regen power capacity and the battery's charge acceptance.")
            st.write(f"**Regenerative energy recovered in this cycle:** {summary['regen_recovered_wh']:.1f} Wh")
            st.write("Regen provides significant braking torque, reducing the reliance on friction brakes. This is especially useful in urban 'stop-and-go' driving, where frequent braking allows for more energy recovery.")

        with tab3:
            st.header("Powertrain Selector & Recommendations")
            desired_range_km = st.number_input("Desired real-world range (km)", value=50.0)
            recs, tiers = recommend_powertrain(summary, desired_range_km=desired_range_km)

            st.subheader("Recommended Specifications")
            st.markdown("Based on the simulation and a safety margin, here are the recommended component specifications:")
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                st.write(f"**Motor Peak Power:** {recs['suggested_motor_peak_w']:.0f} W")
                st.write(f"**Motor Torque:** {recs['suggested_motor_torque_nm']:.1f} Nm")
            with col_rec2:
                if recs['suggested_battery_wh']:
                    st.write(f"**Battery Capacity:** {recs['suggested_battery_wh']:.0f} Wh")

            st.subheader("Candidate Powertrain Tiers")
            st.markdown("Compare different component tiers to see trade-offs between performance and cost.")
            tier_df = pd.DataFrame(tiers)
            st.dataframe(tier_df)

            tier_choice = st.selectbox("Select a tier to evaluate:", [t['name'] for t in tiers])

            if st.button("Simulate with this Tier"):
                chosen = next(t for t in tiers if t['name'] == tier_choice)
                chosen_motor_w = chosen['motor_peak_w']
                chosen_batt_wh = chosen['battery_wh'] if chosen['battery_wh'] else battery_wh

                res_df_tier, summary_tier = run_simulation(
                    res_df[['time_s', 'speed_kmh']], mass, Cd, Cr, Af, wheel_r,
                    battery_wh=chosen_batt_wh, soc0=soc0,
                    battery_max_charge_w=regen_cap_w, battery_max_discharge_w=chosen_motor_w,
                    regen_power_cap_w=regen_cap_w, motor_max_power_w=chosen_motor_w,
                    motor_max_torque_nm=recs['suggested_motor_torque_nm'], motor_knee_rpm=motor_knee_rpm,
                    ambient_rho=DEFAULT_RHO, slope_deg=0.0, gear_ratio=gear_ratio
                )

                st.success(f"Simulation for the **{tier_choice}** tier complete.")
                st.write(f"**Avg. Wh/km:** {summary_tier['wh_per_km']:.1f} Wh/km")
                st.write(f"**Est. Range:** {chosen_batt_wh / summary_tier['wh_per_km']:.0f} km")
                st.plotly_chart(plot_power_time(res_df_tier))
                st.plotly_chart(plot_soc_time(res_df_tier))
    else:
        st.info("Please adjust vehicle parameters and click **Run / Re-run simulation** to begin.")
        st.write("Drive cycle preview (first 60 rows):")
        st.dataframe(default_hwfet_cycle().head(60))

# Entrypoint
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_streamlit_app()
    else:
        print("To run the Streamlit app, make sure Streamlit is installed: pip install streamlit")
        print("Then, run the script from your terminal: streamlit run <your_script_name>.py")
        print("\nRunning a basic command-line simulation demo now...")
        df_cycle = default_hwfet_cycle()
        res, summary = run_simulation(df_cycle,
                                      mass_kg=120, Cd=0.9, Cr=0.008, Af=0.4, wheel_radius_m=0.18,
                                      battery_wh=2000, soc0=0.9,
                                      battery_max_charge_w=3000, battery_max_discharge_w=3000, regen_power_cap_w=3000,
                                      motor_max_power_w=3000, motor_max_torque_nm=10, motor_knee_rpm=3000,
                                      ambient_rho=DEFAULT_RHO, slope_deg=0.0, gear_ratio=1.0)
        print(f"Distance: {summary['distance_km']:.2f} km | Energy Wh: {summary['total_energy_wh']:.1f} | Wh/km: {summary['wh_per_km']:.1f}")
        print(f"Peak Power: {summary['peak_elec_W']:.0f} W")

def run_streamlit_app():
    st.set_page_config(layout='wide', page_title='E-Scooter & E-Bike Powertrain Designer')
    st.title("🛴 E-Scooter Powertrain Designer")
    st.write("Simulate performance and select optimal components based on your drive cycles.")

    if 'results' not in st.session_state:
        st.session_state.results = {}

    with st.sidebar:
        st.header("Vehicle Parameters")
        st.markdown("_Input your vehicle's physical properties._")
        defaults = {
            'Electric scooter/trike': {'mass': 120, 'Cd': 0.9, 'Cr': 0.008, 'Af': 0.4, 'wheel_r': 0.18, 'battery_wh': 2000, 'motor_w': 3000},
            'Electric bicycle': {'mass': 100, 'Cd': 1.0, 'Cr': 0.005, 'Af': 0.5, 'wheel_r': 0.34, 'battery_wh': 720, 'motor_w': 500},
            'Electric motorcycle': {'mass': 250, 'Cd': 0.6, 'Cr': 0.012, 'Af': 0.6, 'wheel_r': 0.32, 'battery_wh': 12000, 'motor_w': 20000},
            'Electric car': {'mass': 1500, 'Cd': 0.28, 'Cr': 0.01, 'Af': 2.2, 'wheel_r': 0.30, 'battery_wh': 60000, 'motor_w': 60000}
        }
        ev_type = st.selectbox("Select EV Type for default parameters", list(defaults.keys()))
        d = defaults[ev_type]

        mass = st.number_input("Vehicle mass (kg)", value=float(d['mass']), min_value=20.0, help="Vehicle + Rider mass")
        Cd = st.number_input("Drag coefficient (Cd)", value=float(d['Cd']), format="%.3f")
        Cr = st.number_input("Rolling resistance (Cr)", value=float(d['Cr']), format="%.4f")
        Af = st.number_input("Frontal area ($m^2$)", value=float(d['Af']), format="%.3f")
        wheel_r = st.number_input("Wheel radius (m)", value=float(d['wheel_r']), format="%.3f")
        gear_ratio = st.number_input("Gear ratio (motor:wheel)", value=1.0, format="%.3f")
        rho_air = st.number_input("Air density (kg/m^3)", value=DEFAULT_RHO)
        slope_deg = st.number_input("Constant slope during cycle (deg)", value=0.0, min_value=-15.0, max_value=15.0, step=0.5)

        st.markdown("---")
        st.header("Motor & Battery Specs")
        st.markdown("_Adjust these values to see their impact._")
        battery_wh = st.number_input("Battery capacity (Wh) - 0 to auto-estimate", value=float(d['battery_wh']), min_value=0.0, step=100.0)
        soc0 = st.slider("Initial SOC (fraction)", 0.0, 1.0, 0.9, 0.01)
        motor_max_power_w = st.number_input("Motor peak power (W)", value=float(d['motor_w']))
        motor_max_torque_nm = st.number_input("Motor torque @ zero speed (Nm)", value=20.0, help="Important for acceleration")
        motor_knee_rpm = st.number_input("Motor knee RPM", value=3000.0, help="Speed at which torque begins to drop")
        regen_cap_w = st.number_input("Max regen power cap (W)", value=3000.0)

        st.markdown("---")
        st.header("Drive Cycles")
        st.markdown("_Upload multiple cycles to compare performance._")
        uploaded_files = st.file_uploader("Upload CSV or XLSX files", type=['csv', 'xlsx'], accept_multiple_files=True)
        use_default = st.checkbox("Use built-in HWFET-like cycle", value=True)

        run_sim_button = st.button("Run / Re-run simulation")

    if run_sim_button:
        st.session_state.results = {}
        df_cycles = {}
        if use_default:
            df_cycles['Default Cycle'] = default_hwfet_cycle()

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                try:
                    df_cycles[file_name] = load_cycle_from_bytes(uploaded_file)
                except Exception as e:
                    st.error(f"Error processing file '{file_name}': {e}")

        if not df_cycles:
            st.warning("Please select a default cycle or upload a file to run the simulation.")
        else:
            for name, df_cycle in df_cycles.items():
                if battery_wh <= 0:
                    temp_batt = 1000.0
                    qres, qsum = run_simulation(df_cycle, mass, Cd, Cr, Af, wheel_r, temp_batt, soc0, regen_cap_w, motor_max_power_w, regen_cap_w, motor_max_power_w, motor_max_torque_nm, motor_knee_rpm, rho_air, slope_deg, gear_ratio)
                    whpkm = qsum['wh_per_km'] if qsum['wh_per_km'] and not np.isnan(qsum['wh_per_km']) else 100.0
                    target_range = 150 if ev_type == 'Electric car' else (120 if ev_type == 'Electric motorcycle' else (60 if ev_type == 'Electric bicycle' else 50))
                    auto_battery_wh = max(200.0, whpkm * target_range)
                else:
                    auto_battery_wh = battery_wh

                res_df, summary = run_simulation(
                    df_cycle, mass, Cd, Cr, Af, wheel_r, auto_battery_wh, soc0, regen_cap_w, motor_max_power_w,
                    regen_cap_w, motor_max_power_w, motor_max_torque_nm, motor_knee_rpm, rho_air, slope_deg, gear_ratio
                )
                st.session_state.results[name] = (res_df, summary)
            st.success("All selected cycles have been simulated.")

    st.header("Simulation Results")

    if st.session_state.results:
        cycle_names = list(st.session_state.results.keys())
        selected_cycle = st.selectbox("Select a drive cycle to view:", cycle_names)

        res_df, summary = st.session_state.results[selected_cycle]

        # New tab structure
        tab1, tab2, tab3 = st.tabs(["📊 Simulation Summary & Plots", "📝 Detailed Calculations", "⚙️ Powertrain Selector"])

        with tab1:
            st.subheader(f"Summary for **{selected_cycle}**")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("Distance", f"{summary['distance_km']:.1f} km")
            with col2: st.metric("Energy Consumption", f"{summary['total_energy_wh']:.1f} Wh")
            with col3: st.metric("Avg. Wh/km", f"{summary['wh_per_km']:.1f} Wh/km")
            with col4: st.metric("Peak Electrical Power", f"{summary['peak_elec_W']:.0f} W")
            with col5: st.metric("Est. Range", f"{summary['battery_wh']/summary['wh_per_km']:.0f} km")

            st.markdown("---")
            st.subheader("Performance Graphs")
            graphs_per_row = 2

            graph_funcs = [
                plot_speed_time, plot_tractive_force, plot_power_time, plot_soc_time,
                plot_force_breakdown, plot_energy_consumption, plot_motor_torque_speed,
                plot_power_demand_hist, plot_accel_vs_speed, plot_regen_power_time
            ]

            for i in range(len(graph_funcs)):
                if i % graphs_per_row == 0:
                    cols = st.columns(graphs_per_row)

                with cols[i % graphs_per_row]:
                    fig = graph_funcs[i](res_df)
                    st.plotly_chart(fig, use_container_width=True)
                    plot_name = fig.layout.title.text
                    insp = insight_block_for_plot(plot_name, res_df, summary)
                    st.markdown(f"**Observations**")
                    for o in insp['observations']: st.write("-", o)
                    st.markdown(f"**Recommendations**")
                    for r in insp['recommendations']: st.write("-", r)
                    st.markdown(f"**Conclusion**")
                    for c in insp['conclusions']: st.write("-", c)

        with tab2:
            st.subheader("Simulation Calculations & Metrics")
            st.write("Below are the key calculated values from the simulation.")

            st.markdown("---")
            st.subheader("Summary Metrics")
            st.write(f"- **Trip distance:** {summary['distance_km']:.2f} km")
            st.write(f"- **Net energy used:** {summary['total_energy_wh']:.1f} Wh")
            st.write(f"- **Average consumption:** {summary['wh_per_km']:.1f} Wh/km")
            st.write(f"- **Estimated range on this cycle:** {summary['battery_wh']/summary['wh_per_km']:.1f} km")
            st.write(f"- **Peak electrical draw:** {summary['peak_elec_W']:.0f} W")
            st.write(f"- **Peak motor torque:** {summary['peak_torque_Nm']:.1f} Nm")
            st.write(f"- **Regenerative energy recovered:** {summary['regen_recovered_wh']:.1f} Wh")

            st.markdown("---")
            st.subheader("Force Breakdown")
            st.write("This table shows the contribution of each force to the total tractive force at key points in the drive cycle.")
            summary_points = res_df.iloc[[0, res_df['v_kmh'].idxmax(), res_df['P_elec_W'].idxmax(), -1]]
            st.dataframe(summary_points[['time_s', 'v_kmh', 'F_aero_N', 'F_roll_N', 'F_slope_N', 'F_iner_N', 'F_trac_req_N']].style.format({
                'time_s': '{:.1f}', 'v_kmh': '{:.1f}', 'F_aero_N': '{:.0f}', 'F_roll_N': '{:.0f}', 'F_slope_N': '{:.0f}', 'F_iner_N': '{:.0f}', 'F_trac_req_N': '{:.0f}'
            }))

            st.markdown("---")
            st.subheader("Vehicle Performance on Slopes")
            st.write("This table shows the required motor power and torque to maintain a constant speed on various slopes.")
            cruise_speed = st.number_input("Cruise speed for slope table (km/h)", value=30.0)
            slope_df = slope_performance_table(mass, Cd, Cr, Af, rho_air, wheel_r, gear_ratio, 0.9, 0.6, battery_wh, [cruise_speed], list(range(0, 16)))
            st.dataframe(slope_df.style.format({'F_trac_N':'{:.1f}','torque_wheel_Nm':'{:.1f}','P_wheel_W':'{:.0f}','P_elec_W':'{:.0f}','Wh_per_km':'{:.2f}','est_range_km':'{:.0f}'}))

        with tab3:
            st.header("Powertrain Selector & Recommendations")
            st.write("Use the simulation results to help you select an appropriate motor and battery for your application.")

            desired_range_km = st.number_input("Desired real-world range (km)", value=50.0)
            recs, tiers = recommend_powertrain(summary, desired_range_km=desired_range_km)

            st.subheader("Recommended Specifications")
            st.markdown("Based on the simulation and a safety margin, here are the recommended component specifications:")
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                st.write(f"**Motor Peak Power:** {recs['suggested_motor_peak_w']:.0f} W")
                st.write(f"**Motor Torque:** {recs['suggested_motor_torque_nm']:.1f} Nm")
            with col_rec2:
                if recs['suggested_battery_wh']:
                    st.write(f"**Battery Capacity:** {recs['suggested_battery_wh']:.0f} Wh")

            st.subheader("Candidate Powertrain Tiers")
            st.markdown("Compare different component tiers to see trade-offs between performance and cost.")
            tier_df = pd.DataFrame(tiers)
            st.dataframe(tier_df.style.format({'motor_peak_w': '{:.0f}', 'battery_wh': '{:.0f}'}))

            tier_choice = st.selectbox("Select a tier to evaluate:", [t['name'] for t in tiers])

            if st.button("Simulate with this Tier"):
                chosen = next(t for t in tiers if t['name'] == tier_choice)
                chosen_motor_w = chosen['motor_peak_w']
                chosen_batt_wh = chosen['battery_wh'] if chosen['battery_wh'] else battery_wh

                res_df_tier, summary_tier = run_simulation(
                    res_df[['time_s', 'speed_kmh']], mass, Cd, Cr, Af, wheel_r,
                    battery_wh=chosen_batt_wh, soc0=soc0,
                    battery_max_charge_w=regen_cap_w, battery_max_discharge_w=chosen_motor_w,
                    regen_power_cap_w=regen_cap_w, motor_max_power_w=chosen_motor_w,
                    motor_max_torque_nm=recs['suggested_motor_torque_nm'], motor_knee_rpm=motor_knee_rpm,
                    ambient_rho=DEFAULT_RHO, slope_deg=0.0, gear_ratio=gear_ratio
                )

                st.success(f"Simulation for the **{tier_choice}** tier complete.")
                st.write(f"**Avg. Wh/km:** {summary_tier['wh_per_km']:.1f} Wh/km")
                st.write(f"**Est. Range:** {chosen_batt_wh / summary_tier['wh_per_km']:.0f} km")
                st.plotly_chart(plot_power_time(res_df_tier))
                st.plotly_chart(plot_soc_time(res_df_tier))
    else:
        st.info("Please adjust vehicle parameters and click **Run / Re-run simulation** to begin.")
        st.write("Drive cycle preview (first 60 rows):")
        st.dataframe(default_hwfet_cycle().head(60))