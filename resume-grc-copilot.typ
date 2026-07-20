// import from template
#import "../template/template.typ": *;
#show: template;

#init(
    name: "孔扬涛",
    // 如不需要头像，可注释掉上面的 pic_path 行，或者将其置空
    // pic_path: "/img/avatar.jpg",
)

#info(
    color: rgb(0, 0, 0),
    // (
    //     icon: "/img/fa/fa-home.svg",
    //     link: "https://zhangsan.io/",
    //     content: "https://zhangsan.io/"
    // ),
    (
        icon: fa_email,
        link: "mailto:Yangtao%20Kong<Phil_Kong818@outlook.com>",
        content: "Phil_Kong818@outlook.com",
    ),
    (
        icon: fa_phone,
        link: "tel:+86 180 6623 5770",
        content: "+86 180 6623 5770",
    ),
    (
        icon: fa_github,
        link: "https://github.com/diveintonull",
        content: "diveintonull"
    ),
)
// // 如果info太长，可以多次调用info实现分行
// #info(
//     color: rgb(0, 0, 0),
//     (
//         icon: fa_github,
//         link: "https://github.com/xxxxxx",
//         content: "xxxxxx"
//     ),
// )

#resume_section("教育经历")

#resume_item(
  "卢森堡大学",
  "硕士 | 网络安全",
  "主修课程：AI安全，移动端安全，微内核系统，密码学，软件和系统漏洞等",
  "2025.09 -- 2027.06（预计）"
)
#resume_item(
  "广西大学（211）",
  "本科 | 信息安全",
  "主修课程：现代密码学，软件安全，Linux 网络编程，机器学习，编译原理，逆向工程等",
  "2020.09 -- 2024.06"
)

#resume_section("专业技能")

#resume_desc(
  "大模型 / Agent 安全",
  [关注智能体（Agent）与大模型安全攻防方向，了解提示词注入（Prompt Injection）、工具调用劫持、越权访问、越狱攻击等智能体核心安全风险；了解主流 Agent 框架（LangChain、AutoGPT 等）的工作机制与攻击面；具备针对智能体的漏洞挖掘、风险验证与异常行为分析思路]
)
#resume_desc(
  "AI 安全基础",
  [了解对抗样本攻击原理（FGSM、PGD 等梯度攻击方法）及对抗扰动的生成与检测；了解模型鲁棒性评估方法；持续跟踪大模型与智能体领域安全与对抗的最新研究方向]
)
#resume_desc(
  "Web 渗透与漏洞挖掘",
  [Web 应用漏洞挖掘与利用，包括 SQL 注入、XSS、SSRF、文件上传、越权访问（IDOR）、业务逻辑缺陷；具备攻防对抗思维与独立完成渗透测试并输出专业报告的能力]
)
#resume_desc(
  "编程与自动化",
  [熟练掌握 Python，可编写安全测试脚本与自动化工具，具备基础数据分析能力；熟悉 Golang，可开发安全工具；了解 C/C++，可阅读代码进行漏洞分析]
)
#resume_desc(
  "外语能力",
  [英语雅思 7.0，可流畅阅读并理解英文安全研究技术报告，撰写英文技术文档]
)

#resume_section("项目经历")

#resume_item(
  "GRC Copilot：证据驱动的法规合规 Agent",
  "Agent / RAG 应用开发",
  "Python · LangGraph · FastAPI · Qdrant · BGE-M3 · MCP · Docker",
  "2026.07"
)
#resume_desc(
  "系统设计",
  [面向法规问答、条款比较与控制差距分析三类场景，构建基于 LangGraph 的受治理 Agent 工作流；设计版本化 Evidence 数据结构、渐进式 Skills 与本地/MCP 工具适配，实现从法规检索、答案生成到引用验证和拒答的完整链路]
)
#resume_desc(
  "检索与调优",
  [使用 BGE-M3、Qdrant 与 Cross-Encoder Reranker 实现父子分块检索；基于 60 条固定评测集完成 8 组检索消融，将 Recall\@5 从固定窗口 Dense 基线的 50.00% 提升至 81.91%，提高 31.91 个百分点]
)
#resume_desc(
  "可信与安全",
  [实现编号引用、版本一致性与证据语义支持校验，对无证据、无效引用及越权合规结论执行拒答或单次修复；将检索文档作为不可信输入进行边界转义，并通过 Prompt Injection 用例验证工具零调用拒答和有限重试行为]
)
#resume_desc(
  "工程交付",
  [基于 FastAPI 与 SSE 实现回答、法规证据和安全 Trace 的流式展示，提供 Docker Compose 本地部署及 Qdrant 首次建索引；编写 300+ 自动化测试覆盖 Agent Graph、RAG、API、MCP、评测指标与流式协议（GitHub：github.com/diveintonull/grc-copilot）]
)

#resume_section("证书资质")

#resume_desc(
  "HTB CPTS",
  [HackTheBox Certified Penetration Testing Specialist · 企业级渗透测试实战认证]
)
#resume_desc(
  "CompTIA Security+",
  [SY0-701]
)
#resume_desc(
  "CISP-PTE",
  [注册信息安全专业人员（渗透测试方向）· 中国信息安全测评中心颁发]
)
