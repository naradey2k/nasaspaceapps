import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import netCDF4

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, LSTM

model = Sequential()
model.add(
    LSTM(
        512,
        batch_input_shape=(None, None, 15),
        stateful=False,
        dropout=0.4,
    )
)
model.add(Dense(2))
model.load_weights('my_model.h5')

def main():
    st.markdown("<h1 style='text-align: center;'>🤖 Predicting Hourly Disturbance Storm-Time </h1>", unsafe_allow_html=True)

    with st.expander("ℹ️ - About App", expanded=True):
        st.write(
                """
        -   The real-time solar wind and interplanetary magnetic field data that you can find on this website come from the DSCOVR satellite. 
                """
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        bt_mean = st.number_input('Insert a mean Bt value', key=1, help='Interplanetary-magnetic-field component magnitude (nT)')
        bt_std = st.number_input('Insert a std Bt value', key=2, help='Interplanetary-magnetic-field component magnitude (nT)')
        temperature_mean = st.number_input('Insert a mean Temperature value', key=3, help='Solar wind ion temperature (Kelvin)')
        temperature_std = st.number_input('Insert a std Temperature value', key=4, help='Solar wind ion temperature (Kelvin)')

    with c2:
        bx_gse_mean = st.number_input('Insert a mean Bx value', key=5, help='Position of the satellite in the X direction of GSE coordinates (km)')
        bx_gse_std = st.number_input('Insert a std Bx value', key=6, help='Position of the satellite in the X direction of GSE coordinates (km)')
        by_gse_mean = st.number_input('Insert a mean By value', key=7, help='Position of the satellite in the Y direction of GSE coordinates (km)')
        by_gse_std = st.number_input('Insert a std By value', key=8, help='Position of the satellite in the Y direction of GSE coordinates (km)')
        bz_gse_mean = st.number_input('Insert a mean Bz value', key=9, help='Position of the satellite in the Z direction of GSE coordinates (km)')
        bz_gse_std = st.number_input('Insert a std Bz value', key=10, help='Position of the satellite in the Z direction of GSE coordinates (km)')

    with c3:
        speed_mean = st.number_input('Insert a mean Speed value', key=11, help='Solar wind bulk speed (km/s)')
        speed_std = st.number_input('Insert a std Speed value', key=12, help='Solar wind bulk speed (km/s)')
        density_mean = st.number_input('Insert a mean Density value', key=13, help='Solar wind proton density (N/cm^3)')
        density_std = st.number_input('Insert a std Density value', key=14, help='Solar wind proton density (N/cm^3)')
        smoothed_ssn = st.number_input('Insert a Sunspot value', key=15)

    
    # col1, col2 = st.columns(2)

    # with col2:
    comp = st.button('Start Predicting', use_container_width=True)

    if comp:
        data = np.array([bt_mean, bt_std, temperature_mean, temperature_std, bx_gse_mean, bx_gse_std, by_gse_mean, by_gse_std, bz_gse_mean, bz_gse_std, speed_mean, speed_std, density_mean, density_std, smoothed_ssn])
        ss_data = (data - data.mean()) / data.std()
        
        preds = model.predict(ss_data[np.newaxis, np.newaxis, :])
        current_hour = preds[0, 0]
        next_hour = preds[0, 1]

        st.info(f'Disturbance Storm-Time for current hour is {current_hour}')
        st.info(f'Disturbance Storm-Time for next hour is {next_hour}')

    st.write("""---""")

    st.markdown("<h3 style='text-align: center;'>🤖 Calculating Norm Coeficient and Entropy before and after Solar Flare </h3>", unsafe_allow_html=True)


    uploaded_file = st.file_uploader("Upload NC file: ", type=["nc"])           

    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.nc')
        temp_file.write(uploaded_file.getbuffer())

        file2read = netCDF4.Dataset(temp_file.name, 'r')

        secs = file2read.variables['time'][:]
        xrsb = file2read.variables['xrsb_flux'][:]

        ot = st.number_input('Insert a start seconds', key=16, value=1500)
        do = st.number_input('Insert a final seconds', key=17, value=15000)

        ot5=ot-300; do5=ot; # 5 min do flare
        ot5n=do; do5n=do+300 # 5 posle flare

        t5 = secs[ot5:do5]
        y5 = xrsb[ot5:do5]
        y5 = y5/max(y5)
        h = 0.001
        mxt0 = max(t5);    
        mnt0 = min(t5)

        Tt = np.arange(0, 1 + h, h)

        # Compute 'Txx' using the sine and arcsine functions
        Txx = np.arcsin(np.sin(np.pi * Tt))

        # Calculate the standard deviation 'vt' of 'Txx'
        vt = np.std(Txx)

        # Normalize 'Txx' by dividing by 'vt' to get 'Tx'
        Tx = Txx / vt

        # Find the maximum and minimum values of 'Tx'
        mxTx = np.max(Tx)
        mnTx = np.min(Tx)

        # Create an array 'xt' from 'mnTx' to 'mxTx' with step 'h'
        xt = np.arange(mnTx, mxTx + h, h)

        # Compute the histogram 'nt' of 'Tx' using 'xt'
        nt, _ = np.histogram(Tx, bins=xt)

        # Compute the sum of 'nt'
        ntmx = np.sum(nt)

        # Calculate 'pt' as nonzeros(nt/ntmx)
        pt = nt[nt != 0] / ntmx

        # Calculate entropy 'Str' using the provided formula
        Str = -np.sum(pt * np.log(pt))

        x = np.arange(min(y5), max(y5) + h, h)
        n, _ = np.histogram(y5, bins=x)
        nmx = np.sum(n)

        p = n[n != 0] / nmx

        size_p = p.size

        # Compute entropy 'Ss' using the formula
        Ss = -np.sum(p * np.log(p))


        # Calculate 'SdoSF'
        SdoSF = Ss / Str

        st.info(f'Entropy before 5 minutes of Solar Flare - {SdoSF}')

        t5 = secs[ot5n:do5n]
        y5 = xrsb[ot5n:do5n]
        y5 = y5/max(y5)
        h = 0.001
        mxt0 = max(t5);    
        mnt0 = min(t5)

        Tt = np.arange(0, 1 + h, h)

        # Compute 'Txx' using the sine and arcsine functions
        Txx = np.arcsin(np.sin(np.pi * Tt))

        # Calculate the standard deviation 'vt' of 'Txx'
        vt = np.std(Txx)

        # Normalize 'Txx' by dividing by 'vt' to get 'Tx'
        Tx = Txx / vt

        # Find the maximum and minimum values of 'Tx'
        mxTx = np.max(Tx)
        mnTx = np.min(Tx)

        # Create an array 'xt' from 'mnTx' to 'mxTx' with step 'h'
        xt = np.arange(mnTx, mxTx + h, h)

        # Compute the histogram 'nt' of 'Tx' using 'xt'
        nt, _ = np.histogram(Tx, bins=xt)

        # Compute the sum of 'nt'
        ntmx = np.sum(nt)

        # Calculate 'pt' as nonzeros(nt/ntmx)
        pt = nt[nt != 0] / ntmx

        # Calculate entropy 'Str' using the provided formula
        Str = -np.sum(pt * np.log(pt))

        x = np.arange(min(y5), max(y5) + h, h)
        n, _ = np.histogram(y5, bins=x)
        nmx = np.sum(n)

        p = n[n != 0] / nmx

        size_p = p.size

        # Compute entropy 'Ss' using the formula
        Ss = -np.sum(p * np.log(p))


        # Calculate 'SdoSF'
        SdoSF = Ss / Str

        st.info(f'Entropy after 5 minutes of Solar Flare - {SdoSF}')


if __name__ == "__main__":
    main()