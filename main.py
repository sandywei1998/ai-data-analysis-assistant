import pandas as pd
import streamlit as st
from utils import dataframe_agent

def create_chart(input_data, chart_type):
    df_data = pd.DataFrame(input_data["data"], columns = input_data["columns"])
    df_data.set_index(input_data["columns"][0], inplace = True)
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        st.line_chart(df_data)
    elif chart_type == "scatter":
        st.scatter(df_data)

st.title("💡 数据分析智能工具")

with st.sidebar:
    api_key = st.text_input("请输入API密钥", type = "password")
    st.markdown("[获取API KEY](*)")

data = st.file_uploader("上传你的数据文件（CSV格式）：", type = "csv")
if data:
    st.session_state["df"] = pd.read_csv(data)
    with st.expander("原始数据"):
        st.dataframe(st.session_state["df"])

query = st.text_area("请输入你关于以上表格的问题，或者数据提取请求，或者可视化要求（仅支持散点图、折线图、条形图）：")
button = st.button("生成回答")

if button and not api_key:
    st.info("请输入你的API密钥")
if button and "df" not in st.session_state:
    st.info("请先上传数据文件")
if button and api_key and "df" in st.session_state:
    with st.spinner("AI正在思考中，请稍等..."):
        responses_dict = dataframe_agent(api_key, st.session_state["df"], query)
        if "answer" in responses_dict:
            st.write(responses_dict["answer"])
        if "table" in responses_dict:
            st.table(pd.DataFrame(responses_dict["table"]["data"],
                                  columns = responses_dict["table"]["columns"]))
        if "bar" in responses_dict:
            print("AI返回的bar数据：", responses_dict["bar"])
            create_chart(responses_dict["bar"], "bar")
        if "line" in responses_dict:
            create_chart(responses_dict["line"], "line")
        if "scatter" in responses_dict:
            create_chart(responses_dict["scatter"], "scatter")