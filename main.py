import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="EV Drive Cycle Simulator", layout="wide")

# --------------------------- Utilities ---------------------------
MPH_TO_MS = 0.44704
KMH_TO_MS = 1/3.6

@st.cache_data
def default_hwfet_cycle():
    # Synthetic short HWFET-like cycle (time s, speed km/h)
    t = np.arange(0, 600, 1)
    # create a realistic-ish speed trace with accelerations and decels
    speed = 20 + 30*np.sin(2*np.pi*t/180) + 10*np.sin(2*np.pi*t/40)
    speed = np.clip(speed, 0, 120)
    return pd.DataFrame({'time_s': t, 'speed_kmh': speed})


def load_drive_cycle(uploaded_file):
    if uploaded_file is None:
        return default_hwfet_cycle()
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        txt = StringIO(uploaded_file.getvalue().decode('utf-8'))
        df = pd.read_csv(txt)
    # try to find columns
    if 'time' in df.columns and 'speed' in df.columns:
        df = df.rename(columns={'time': 'time_s', 'speed': 'speed_kmh'})
    elif 'time_s' in df.columns and 'speed_kmh' in df.columns:
        pass
    else:
        # try first two columns
        df = df.iloc[:, :2]
        df.columns = ['time_s', 'speed_kmh']
    return df


def compute_simulation(df, params):
    # Inputs
    m = params['mass']
    Cd = params['Cd']
    Cr = params['Cr']
    Af = params['Af']
    rho = params['rho_air']
    g = params['g']
    wheel_r = params['wheel_radius']
    motor_eff = params['motor_eff']
    regen_eff = params['regen_eff']
    battery_volt = params['battery_voltage']
    battery_ah = params['battery_ah']
    initial_soc = params['initial_soc']

    # Prepare arrays
    df = df.copy().reset_index(drop=True)
    dt = df['time_s'].diff().fillna(df['time_s'].iloc[1] - df['time_s'].iloc[0])
    v_ms = df['speed_kmh'] * KMH_TO_MS

    # acceleration
    dv = v_ms.diff().fillna(0)
    a = dv / dt

    # aerodynamic force
    F_aero = 0.5 * rho * Cd * Af * v_ms**2
    # rolling resistance (approx)
    F_roll = m * g * Cr * np.sign(v_ms)  # sign to allow direction
    # gravitational component (for slope zero here)
    slope_rad = 0.0
    F_slope = m * g * np.sin(slope_rad)

    # inertial force
    F_iner = m * a

    # tractive force required at wheel (positive = drive, negative = braking)
    F_trac = F_aero + F_roll + F_slope + F_iner

    # motor torque at wheel
    torque_wheel = F_trac * wheel_r

    # mechanical power at wheel (W)
    P_wheel = F_trac * v_ms

    # electrical power to/from battery (account for motor efficiency)
    P_elec = np.where(P_wheel >= 0, P_wheel / motor_eff, P_wheel * regen_eff)
    # when braking, P_wheel negative; regen_eff fraction returned to battery (positive recovered)

    # integrate energy and SOC
    energy_flow_wh = np.cumsum(P_elec * dt) / 3600.0  # Wh (can be negative for net discharge)
    # battery capacity in Wh
    if battery_ah is None or battery_ah <= 0 or battery_volt is None or battery_volt <= 0:
        battery_wh = params['estimated_battery_wh']
    else:
        battery_wh = battery_volt * battery_ah

    soc = initial_soc - energy_flow_wh / battery_wh
    soc = np.clip(soc, 0, 1)

    df['dt_s'] = dt
    df['v_ms'] = v_ms
    df['v_kmh'] = df['speed_kmh']
    df['acc_m_s2'] = a
    df['F_aero_N'] = F_aero
    df['F_roll_N'] = F_roll
    df['F_iner_N'] = F_iner
    df['F_trac_N'] = F_trac
    df['torque_wheel_Nm'] = torque_wheel
    df['P_wheel_W'] = P_wheel
    df['P_elec_W'] = P_elec
    df['energy_flow_wh'] = energy_flow_wh
    df['soc'] = soc

    results = {
        'df': df,
        'battery_wh': battery_wh,
        'total_energy_wh': energy_flow_wh.iloc[-1],
        'energy_consumption_Wh_per_km': None,
        'peak_power_W': np.nanmax(P_wheel),
        'peak_torque_Nm': np.nanmax(np.abs(torque_wheel)),
    }

    # compute trip distance
    distance_m = np.sum(v_ms * dt)
    distance_km = distance_m / 1000.0
    if distance_km > 0:
        results['energy_consumption_Wh_per_km'] = results['total_energy_wh'] / distance_km

    return results


def slope_performance_table(cruise_speed_kmh, slopes_deg, params):
    v_ms = cruise_speed_kmh * KMH_TO_MS
    m = params['mass']
    Cd = params['Cd']
    Cr = params['Cr']
    Af = params['Af']
    rho = params['rho_air']
    g = params['g']
    wheel_r = params['wheel_radius']
    motor_eff = params['motor_eff']

    rows = []
    for slope in slopes_deg:
        theta = np.deg2rad(slope)
        F_aero = 0.5 * rho * Cd * Af * v_ms**2
        F_roll = m * g * Cr
        F_slope = m * g * np.sin(theta)
        F_iner = 0.0
        F_trac = F_aero + F_roll + F_slope + F_iner
        P_wheel = F_trac * v_ms
        torque = F_trac * wheel_r
        P_elec = P_wheel / motor_eff if P_wheel >= 0 else P_wheel * params['regen_eff']
        rows.append({'slope_deg': slope, 'F_trac_N': F_trac, 'torque_Nm': torque, 'P_wheel_W': P_wheel, 'P_elec_W': P_elec})
    return pd.DataFrame(rows)


def generate_textual_insights(results):
    df = results['df']
    battery_wh = results['battery_wh']
    total_energy_wh = results['total_energy_wh']
    energy_per_km = results['energy_consumption_Wh_per_km']

    peak_power = results['peak_power_W']
    peak_torque = results['peak_torque_Nm']

    # observations
    obs = []
    obs.append(f"Trip distance: {round(np.sum(df['v_ms']*df['dt_s'])/1000,3)} km")
    obs.append(f"Total energy (net) during cycle: {total_energy_wh:.1f} Wh")
    if energy_per_km is not None:
        obs.append(f"Average consumption: {energy_per_km:.1f} Wh/km")
    obs.append(f"Peak mechanical power at wheels: {peak_power:.0f} W")
    obs.append(f"Peak torque at wheel: {peak_torque:.1f} Nm")

    # recommendations
    rec = []
    if peak_power > 20000:
        rec.append('Consider a high-power motor (>= 20 kW) or gearbox for heavy loads/high speed bursts.')
    elif peak_power > 5000:
        rec.append('Mid-range motor (5-20 kW) should be adequate for this cycle.')
    else:
        rec.append('Small motor (<5 kW) appears sufficient for this duty cycle.')

    if energy_per_km is not None:
        est_range_km = battery_wh / energy_per_km if energy_per_km>0 else np.nan
        rec.append(f'Estimated real-world range on this cycle (from battery): {est_range_km:.0f} km')

    # conclusions
    concl = []
    regen_recovered_wh = -np.sum(df.loc[df['P_elec_W']<0, 'P_elec_W']*df.loc[df['P_elec_W']<0,'dt_s'])/3600.0
    concl.append(f"Regenerative braking recovered approx {regen_recovered_wh:.1f} Wh in this cycle (depends on regen efficiency).")
    concl.append('Design trade-offs: increasing battery size raises range but also mass which increases consumption; optimizing aerodynamics and rolling resistance gives large gains at higher speeds.')

    return {'observations': obs, 'recommendations': rec, 'conclusions': concl}


# --------------------------- Streamlit UI ---------------------------
st.title('EV Drive Cycle Simulator (Streamlit + Plotly)')

with st.sidebar:
    st.header('Vehicle & Simulation Parameters')
    ev_type = st.selectbox('Select EV type', ['Electric car', 'Electric motorcycle', 'Electric bicycle', 'Electric trike/scooter'])

    # defaults based on EV type
    defaults = {
        'Electric car': {'mass': 1500, 'Cd': 0.28, 'Cr': 0.01, 'Af': 2.2, 'wheel_r': 0.3},
        'Electric motorcycle': {'mass': 250, 'Cd': 0.6, 'Cr': 0.012, 'Af': 0.6, 'wheel_r': 0.32},
        'Electric bicycle': {'mass': 100, 'Cd': 1.0, 'Cr': 0.005, 'Af': 0.5, 'wheel_r': 0.34},
        'Electric trike/scooter': {'mass': 120, 'Cd': 0.9, 'Cr': 0.008, 'Af': 0.4, 'wheel_r': 0.18},
    }
    d = defaults[ev_type]

    mass = st.number_input('Vehicle mass (kg)', value=float(d['mass']), min_value=20.0)
    Cd = st.number_input('Drag coefficient (Cd)', value=float(d['Cd']), format="%.3f")
    Cr = st.number_input('Rolling resistance coefficient (Cr)', value=float(d['Cr']), format="%.4f")
    Af = st.number_input('Frontal area (m^2)', value=float(d['Af']), format="%.3f")
    wheel_radius = st.number_input('Wheel radius (m)', value=float(d['wheel_r']), format="%.3f")

    st.subheader('Battery / Motor')
    battery_voltage = st.number_input('Battery pack nominal voltage (V) (leave 0 to estimate)', value=0.0)
    battery_ah = st.number_input('Battery capacity (Ah) (leave 0 to estimate)', value=0.0)
    initial_soc = st.slider('Initial State of Charge (fraction)', 0.0, 1.0, 0.9, 0.01)
    motor_eff = st.slider('Motor+inverter efficiency (fraction)', 0.5, 0.99, 0.9, 0.01)
    regen_eff = st.slider('Regen efficiency (fraction of braking power returned)', 0.0, 0.9, 0.5, 0.01)

    st.subheader('Environment & sim')
    rho_air = st.number_input('Air density (kg/m^3)', value=1.2)
    g = st.number_input('Gravity (m/s^2)', value=9.81)

    st.subheader('Drive cycle')
    uploaded = st.file_uploader('Upload CSV drive cycle (time_s, speed_kmh) - optional', type=['csv'])
    use_default = st.checkbox('Use default synthetic HWFET-like cycle', value=True)
    regen_on = st.checkbox('Enable regenerative braking in sim', value=True)

    run_sim = st.button('Run simulation')

# Prepare params
params = {
    'mass': mass,
    'Cd': Cd,
    'Cr': Cr,
    'Af': Af,
    'rho_air': rho_air,
    'g': g,
    'wheel_radius': wheel_radius,
    'motor_eff': motor_eff,
    'regen_eff': regen_eff if regen_on else 0.0,
    'battery_voltage': battery_voltage if battery_voltage>0 else None,
    'battery_ah': battery_ah if battery_ah>0 else None,
    'initial_soc': initial_soc,
}

# load drive cycle
if use_default or uploaded is None:
    df_cycle = default_hwfet_cycle()
else:
    df_cycle = load_drive_cycle(uploaded)

# estimate battery if not provided
# run a quick zero-battery sim to estimate energy demand per km and then set estimated_battery_wh
quick_params = params.copy()
quick_params['battery_voltage'] = 1.0
quick_params['battery_ah'] = 1.0
quick_params['estimated_battery_wh'] = 50000.0  # placeholder
quick_res = compute_simulation(df_cycle, quick_params)
energy_per_km = quick_res['energy_consumption_Wh_per_km']
if energy_per_km is None or np.isnan(energy_per_km) or energy_per_km<=0:
    estimated_wh = 10000
else:
    # default target range 150 km for cars, scaled by vehicle type
    default_range_km = 150 if ev_type=='Electric car' else (120 if ev_type=='Electric motorcycle' else (60 if ev_type=='Electric bicycle' else 50))
    estimated_wh = energy_per_km * default_range_km
params['estimated_battery_wh'] = estimated_wh

if run_sim:
    results = compute_simulation(df_cycle, params)
    df = results['df']

    # create plots
    st.header('Results & Plots')
    col1, col2 = st.columns([1,1])

    with col1:
        fig_speed = px.line(df, x='time_s', y='v_kmh', title='Speed vs Time', labels={'time_s':'Time (s)','v_kmh':'Speed (km/h)'})
        st.plotly_chart(fig_speed, use_container_width=True)

        fig_force = px.line(df, x='time_s', y='F_trac_N', title='Tractive Force vs Time', labels={'F_trac_N':'Tractive Force (N)'})
        st.plotly_chart(fig_force, use_container_width=True)

    with col2:
        # Motor/mechanical/electrical power
        pfig = go.Figure()
        pfig.add_trace(go.Scatter(x=df['time_s'], y=df['P_wheel_W'], name='Wheel Power (W)'))
        pfig.add_trace(go.Scatter(x=df['time_s'], y=df['P_elec_W'], name='Electrical Power (W)'))
        pfig.update_layout(title='Power vs Time', xaxis_title='Time (s)', yaxis_title='Power (W)')
        st.plotly_chart(pfig, use_container_width=True)

        fig_soc = px.line(df, x='time_s', y='soc', title='Battery SOC vs Time', labels={'soc':'State of Charge (fraction)'})
        st.plotly_chart(fig_soc, use_container_width=True)

    # Slope performance
    st.subheader('Vehicle performance on slopes (steady cruise)')
    cruise_speed = st.number_input('Cruise speed for slope table (km/h)', value=50.0)
    slopes = list(range(0,16))
    slope_df = slope_performance_table(cruise_speed, slopes, params)
    st.dataframe(slope_df.style.format({'F_trac_N':'{:.1f}','torque_Nm':'{:.1f}','P_wheel_W':'{:.0f}','P_elec_W':'{:.0f}'}))

    # Range vs speed on slopes: for each slope, compute energy consumption per km across a speed range
    st.subheader('Range vs Speed on different slopes')
    speeds = np.arange(10,121,5)
    range_table = []
    for slope in [0,5,10,15]:
        params_temp = params.copy()
        rows = []
        for s in speeds:
            row = slope_performance_table(s, [slope], params_temp).iloc[0]
            # Wh per km = (P_elec_W / v_ms) * 3600? Actually energy per km = P_elec (W) / v (m/s) * 1000 m = P_elec / v * 1000 / 3600? Simpler: Wh/km = P_elec (W) / (v_mps) * (1/1000) * (1/3600) ??? Let's compute directly:
            v_m = s * KMH_TO_MS
            # Energy to travel 1 km = P_elec_W * time_to_travel_1km (s) / 3600 -> Wh
            time_1km = 1000.0 / v_m
            wh_per_km = row['P_elec_W'] * time_1km / 3600.0
            estimated_range_km = params['estimated_battery_wh'] / wh_per_km if wh_per_km>0 else np.nan
            rows.append({'slope': slope, 'speed_kmh': s, 'Wh_per_km': wh_per_km, 'est_range_km': estimated_range_km})
        range_table.append(pd.DataFrame(rows))

    # plot range curves
    rtfig = go.Figure()
    for rdf in range_table:
        slopename = f"slope={int(rdf['slope'].iloc[0])}deg"
        rtfig.add_trace(go.Scatter(x=rdf['speed_kmh'], y=rdf['est_range_km'], mode='lines+markers', name=slopename))
    rtfig.update_layout(title='Estimated Range vs Speed on slopes (using estimated battery size)', xaxis_title='Speed (km/h)', yaxis_title='Range (km)')
    st.plotly_chart(rtfig, use_container_width=True)

    # textual insights
    insights = generate_textual_insights(results)
    st.subheader('Observations')
    for o in insights['observations']:
        st.write('- ' + o)
    st.subheader('Recommendations')
    for r in insights['recommendations']:
        st.write('- ' + r)
    st.subheader('Conclusions')
    for c in insights['conclusions']:
        st.write('- ' + c)

    # allow user to download simulation data
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download simulation CSV', csv, file_name='ev_simulation_output.csv', mime='text/csv')

else:
    st.info('Set parameters in the sidebar and press "Run simulation"')

