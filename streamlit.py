from datetime import datetime
from deep_research_anything.models.event import (
    BaseEvent,
    ErrorEvent,
    ReasoningEvent,
    ResearchStateEvent,
    AgentSelectionEvent,
)
from deep_research_anything.agent.agents.coding.models import CodeExecutionEvent
from deep_research_anything.agent.agents.end.models import (
    GenerateEssayCompleteEvent,
    GenerateEssayStartEvent,
    RefinedKnowledgeEvent,
)
from deep_research_anything.agent.agents.search.models import (
    BatchReadCompleteEvent,
    BatchReadStartEvent,
    BatchSearchCompleteEvent,
    BatchSearchStartEvent,
    NewKnowledgeEvent,
    PageNotAllowedToReadEvent,
)
import streamlit as st
import asyncio
from deep_research_anything.deep_research import Research
import time


def display_knowledge_items(knowledge_items, container):
    if not knowledge_items:
        container.info("No knowledge accumulated yet")
        return

    for i, item in enumerate(knowledge_items):
        container.markdown(f"**Knowledge Point #{i+1}**")
        container.write(item.content)
        container.caption(f"Sources: {', '.join(item.sources)}")
        container.divider()


def display_search_results(pages, container):
    if not pages:
        container.info("No search results yet")
        return

    for i, page in enumerate(pages):
        container.subheader(f"Result #{i+1}: {page.title}")
        container.write("Description:")
        container.write(page.description)
        container.write("Content Preview:")
        container.write(
            page.markdown[:500] + "..." if len(page.markdown) > 500 else page.markdown
        )
        container.caption(f"URL: {page.url}")
        if page.modified_time:
            container.caption(f"Modified Time: {page.modified_time}")
        container.divider()


def display_research_progress(event: BaseEvent, container):
    if isinstance(event, BatchSearchStartEvent):
        container.markdown("### 🔍 Search Started")
        container.info("Searching for:")
        for query in event.query_strings:
            container.markdown(f"- {query}")
        container.caption(f"Sub-goal: {event.sub_goal}")
        container.caption(
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}"
        )

    elif isinstance(event, BatchSearchCompleteEvent):
        container.markdown("### ✅ Search Completed")
        container.success(f"Found {len(event.results)} results")
        container.caption(
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}"
        )
        display_search_results(event.results, container)

    elif isinstance(event, BatchReadStartEvent):
        container.markdown("### 📖 Reading Started")
        container.info("Reading the following articles:")
        for i, page in enumerate(event.pages):
            container.markdown(f"**{i+1}. {page.title}**")
            container.caption(f"URL: [{page.url}]({page.url})")
        container.caption(
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}"
        )

    elif isinstance(event, BatchReadCompleteEvent):
        container.markdown("### ✅ Reading Completed")
        container.success(f"Finished reading: {len(event.pages)} articles")
        container.caption(
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}"
        )

    elif isinstance(event, NewKnowledgeEvent):
        container.markdown("### 💡 New Knowledge Discovered")
        container.success(f"Discovered {len(event.new_items)} new knowledge items")
        container.caption(
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}"
        )
        display_knowledge_items(event.new_items, container)

    elif isinstance(event, RefinedKnowledgeEvent):
        container.markdown("### 🔄 Knowledge Refined")
        container.info("Knowledge has been cross-validated and refined")
        container.caption(
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}"
        )

        col1, col2 = container.columns(2)
        with col1:
            col1.subheader("Original Knowledge")
            display_knowledge_items(event.original_items, col1)
        with col2:
            col2.subheader("Refined Knowledge")
            display_knowledge_items(event.refined_items, col2)

    elif isinstance(event, GenerateEssayStartEvent):
        container.markdown("### 📝 Report Generation Started")
        container.info("Generating final report...")
        container.caption(
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}"
        )

    elif isinstance(event, GenerateEssayCompleteEvent):
        container.markdown("### ✨ Report Generation Completed")
        container.success("Report generated successfully!")
        container.caption(
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}"
        )
        container.markdown("### Final Report")
        container.markdown(event.essay)

    elif isinstance(event, PageNotAllowedToReadEvent):
        container.markdown("### 🚫 Page Not Allowed to Read")
        container.info(
            f"Page {event.page.title} is not allowed to be read because it was published after the news time"
        )
        container.caption(
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}"
        )

    elif isinstance(event, ReasoningEvent):
        container.markdown("### 💭 Reasoning")
        container.write(event.reasoning)
        container.caption(f"Reasoning Step: {event.action}")

    elif isinstance(event, AgentSelectionEvent):
        container.markdown("### 🔧 Agent Selection")
        container.write(f"Selected Agent: {event.agent_selection}")

    elif isinstance(event, CodeExecutionEvent):
        container.markdown("### 🔍 Code Execution")
        container.markdown(
            f"Executed Code:\n```python\n{event.code_result.code_str}\n```"
        )
        container.write(f"Execution Result:")
        container.write(event.code_result.stdout)
        container.write(event.code_result.output)

    elif isinstance(event, ResearchStateEvent):
        container.markdown("### 📚 Research State Summary")
        container.caption(
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}"
        )

        # Display all searched queries
        container.subheader("Search Queries Attempted")
        for i, query in enumerate(event.research_state.searched_queries):
            container.markdown(f"**Query {i+1}:** {query.query_string}")
            container.caption(f"Sub-goal: {query.sub_goal}")
            container.markdown(f"Found {len(query.search_result_pages)} results")

        # Display all collected knowledge
        container.subheader("All Knowledge Collected")
        display_knowledge_items(event.research_state.knowledge, container)
    elif isinstance(event, ErrorEvent):
        container.markdown("### 🚨 Error")
        container.error(event.error)
        container.write(event.traceback)


def main():
    st.set_page_config(layout="wide")
    st.title("🔍 Deep Research Assistant")

    with st.container():
        goal = st.text_area(
            "Research Topic", placeholder="Enter the topic you want to research..."
        )
        research_datetime = st.text_input(
            "Reference Date/Time",
            placeholder="Enter the reference date/time for your research, format: YYYY-MM-DD HH:MM:SS",
            value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        max_steps = st.slider(
            "Maximum Research Steps", min_value=1, max_value=20, value=5
        )
        min_steps = st.slider(
            "Minimum Research Steps", min_value=1, max_value=10, value=2
        )
        start_button = st.button(
            "Start Research",
            disabled=st.session_state.get("research_in_progress", False),
        )

    progress_container = st.container()

    if start_button and goal:
        # Set research in progress flag
        st.session_state["research_in_progress"] = True

        # Display initial information
        progress_container.info(
            "🚀 Research process has started! Please wait while information is being collected..."
        )

        def progress_callback(event: BaseEvent):
            display_research_progress(event, progress_container)
            time.sleep(0.1)  # Small delay to allow UI updates

        result = asyncio.run(
            Research(
                goal=goal,
                research_datetime=research_datetime,
                progress_callback=progress_callback,
            ).run_research(
                max_trajectory_length=max_steps,
                min_trajectory_length=min_steps,
            )
        )

        # Reset research in progress flag
        st.session_state["research_in_progress"] = False


if __name__ == "__main__":
    main()
