import json
import re
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

PROMPT_TEMPLATE = """
你是一位数据分析助手，你的回应内容取决于用户的请求内容。
你必须严格走完ReAct思考步骤，执行代码，**最终在Final Answer输出JSON**，不要提前输出JSON。
1. 对于文字回答的问题，按照这样的格式回答：
   {"answer": "<你的答案写在这里>"}
例如：
   {"answer": "订单量最高的产品ID是'MNWC3-067'"}
2. 如果用户需要一个表格，按照这样的格式回答：
   {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
3. 如果用户的请求适合返回条形图，输出格式为：
   {"bar": {"columns": ["分类字段", "数值字段"], "data": [["分类A", 34], ["分类B", 21], ["分类C", 91]]}}

4. 如果用户的请求适合返回折线图，输出格式为：
   {"line": {"columns": ["X轴字段", "Y轴字段"], "data": [["X值1", 34], ["X值2", 21], ["X值3", 91]]}}

5. 如果用户的请求适合返回散点图，输出格式为：
   {"scatter": {"columns": ["X轴字段", "Y轴字段"], "data": [[10, 20], [15, 30], [20, 25]]}}

图表生成规则：

* 必须根据用户的实际问题和 DataFrame 中的字段，动态选择字段，不能固定使用某几个字段。
* columns 中填写所选字段的实际名称，不能固定为“月份”“销售额”或其他特定字段。
* data 必须是二维列表，每个内层列表代表一条记录，元素数量必须与 columns 的数量相同。
* 条形图：通常使用分类字段作为 X 轴，数值字段作为 Y 轴；必要时先按分类分组、汇总数值。
* 折线图：根据用户的分析目标选择合适的 X 轴和 Y 轴。通常用于展示随时间变化的趋势，但也可根据用户明确的需求使用其他有序的 X 轴。
* 散点图：选择两个有意义的数值字段分别作为 X 轴和 Y 轴，用于分析它们之间的关系。
* 如果需要统计或汇总，必须先执行相应的数据处理，再返回图表数据。
* 只返回用户请求对应的图表类型，不要虚构字段、数据或统计结果。
* 如果数据不适合用户要求的图表，应说明原因，不能强行生成。

注意：我们只支持三种类型的图表："bar", "line" 和 "scatter"。
请将最终输出作为JSON字符串返回。请注意要将"columns"列表和数据列表中的所有字符串都用双引号包围。
例如：{"columns": ["Products", "Orders"], "data": [["32085Lip", 245], ["76439Eye", 178]]}
你要处理的用户请求如下： 
"""

def dataframe_agent(openai_api_key, df, query):
    model = ChatOpenAI(
        model="glm-5",
        openai_api_key=openai_api_key,
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        temperature=0
    )
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        agent_executor_kwargs={},
        verbose=True,
        allow_dangerous_code=True
    )
    prompt = PROMPT_TEMPLATE + query

    response = agent.invoke({"input": prompt})
    output = response["output"].strip()

    # 如果 AI 使用 Markdown 代码块包裹 JSON，就去掉包裹标记
    output = re.sub(r"^```(?:json)?\s*", "", output, flags=re.IGNORECASE)
    output = re.sub(r"\s*```$", "", output)

    # 解析纯 JSON
    response_dict = json.loads(output)

    return response_dict
