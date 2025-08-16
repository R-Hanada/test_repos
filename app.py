import streamlit as st
import pandas as pd
import numpy as np
chart_data = pd.DataFrame( np.random.randn(20, 3), columns=['a', 'b', 'c']
)
st.subheader("折れ線グラフ")
st.line_chart(chart_data)
st.subheader("面グラフ")
st.area_chart(chart_data)
st.subheader("棒グラフ")
st.bar_chart(chart_data)