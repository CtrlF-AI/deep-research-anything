from deep_research_anything.llms import generate_text
import tempfile
import webbrowser
import os


class ReportToHTMLConverter:
    def __init__(self, model=None):
        self.model = model

    def get_html_prompt_zh(self, report: str) -> str:
        return f"""请将以下研究报告转换成一个专业、美观且交互式的HTML页面，风格类似于高端金融或科技分析报告。

请完成以下任务：
1. 内容分析和提炼：
   - 提取报告中的关键数据和统计数字
   - 识别并突出显示主要结论和重点发现
   - 生成简明的要点总结

2. 页面结构和设计：
   - 顶部使用醒目的标题栏，包含主标题和副标题，可使用渐变色背景（优先考虑蓝色系）
   - 采用卡片式布局展示不同类别的信息，带有圆角边框和阴影效果
   - 每个卡片使用图标或标志增强视觉识别
   - 确保布局干净整洁，留有足够留白空间
   - 使用现代化字体和配色方案

3. 数据可视化：
   - 创建专业的交互式图表（使用Chart.js或ECharts）
   - 重要数据使用颜色编码（如上涨用绿色，下跌用红色）
   - 图表应有标题、图例和适当的坐标轴标签
   - 确保图表风格与整体设计协调一致

4. 增强功能和细节：
   - 添加适当的图标（如箭头表示趋势、信息图标等）
   - 实现响应式设计，确保在不同设备上美观展示
   - 添加淡入淡出和渐变效果增强视觉体验
   - 提供暗色/亮色主题切换选项

请确保设计专业且美观，类似于专业金融分析或科技公司的报告页面。返回完整的HTML代码，包含所有必要的CSS样式和JavaScript。

<研究报告>
{report}
</研究报告>
"""

    def get_html_prompt_en(self, report: str) -> str:
        return f"""Convert the following research report into a professional, visually appealing, and interactive HTML page similar to high-end financial or tech analysis reports.

Please complete the following tasks:
1. Content Analysis and Extraction:
   - Extract key data and statistical figures from the report
   - Identify and highlight main conclusions and key findings
   - Generate a concise summary of key points

2. Page Structure and Design:
   - Create a prominent header section with main title and subtitle, preferably with gradient background (blue tones recommended)
   - Use card-based layout with rounded corners and subtle shadows for different information categories
   - Include icons or symbols for each card to enhance visual recognition
   - Ensure clean layout with adequate whitespace
   - Use modern typography and color schemes

3. Data Visualization:
   - Create professional interactive charts (using Chart.js or ECharts)
   - Use color coding for important data (e.g., green for increases, red for decreases)
   - Include titles, legends, and appropriate axis labels for charts
   - Ensure chart styling matches the overall design aesthetic

4. Enhanced Features and Details:
   - Add appropriate icons (arrows for trends, information icons, etc.)
   - Implement responsive design for all devices
   - Include fade-in and gradient effects to enhance visual experience
   - Provide dark/light theme toggle option

Ensure the design is professional and aesthetically pleasing, similar to reports from professional financial analysts or tech companies. Return complete HTML code with all necessary CSS styles and JavaScript.

<research_report>
{report}
</research_report>
"""

    async def generate_html(self, report: str, use_english_prompt: bool = False) -> str:
        prompt = self.get_html_prompt_en(report) if use_english_prompt else self.get_html_prompt_zh(report)
        
        context = []
        html_code = await generate_text(
            model=self.model,
            prompt=prompt,
            context=context
        )
        return html_code
    
    def render_html(self, html_code: str) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
            f.write(html_code)
            temp_path = f.name
        
        # 使用默认浏览器打开HTML文件
        webbrowser.open('file://' + os.path.abspath(temp_path))
        
        print(f"HTML报告已在浏览器中打开，临时文件位置: {temp_path}")


async def convert_report_to_html(report: str, model=None, use_english_prompt: bool = False, render: bool = True) -> str:
    converter = ReportToHTMLConverter(model=model)
    html_code = await converter.generate_html(report, use_english_prompt)
    
    if render:
        converter.render_html(html_code)
    
    return html_code


if __name__ == "__main__":
    import asyncio
    with open("../test/report.txt") as f:
        report = f.read()
    from deep_research_anything.llms import Claude37Sonnet
    asyncio.run(convert_report_to_html(report, model=Claude37Sonnet))
