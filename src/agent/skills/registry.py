"""Explicit local skills exposed to the operation agent.

Knowledge retrieval stays on its ACL-aware graph branch. Discovery alone must
not grant arbitrary newly installed skills access to the operation agent.
"""

OPERATION_SKILLS = (
    "datetime", "calculator", "statistics", "general",
    "data_analysis", "unit_conversion", "text_analysis", "citation_check", "rag_evaluation",
    "paper_writing",
)

RESEARCH_SKILLS = OPERATION_SKILLS[4:]


def operation_skill_instructions() -> str:
    """Expose the research workflows as well as their tool schemas."""
    from .skill_loader import get_skill_loader

    loader = get_skill_loader()
    return "\n\n".join(loader.load_skill(name).prompt for name in RESEARCH_SKILLS)


def selected_operation_skills(names):
    from src.agent.routing.policy import ALLOWED_SKILLS
    from .skill_loader import get_skill_loader
    selected = tuple(dict.fromkeys(names))
    if not selected or any(name not in ALLOWED_SKILLS for name in selected):
        raise ValueError("Unknown or empty operation capabilities")
    loader = get_skill_loader()
    return [loader.load_skill(name) for name in selected]


def get_selected_tools(names):
    return [tool for skill in selected_operation_skills(names) for tool in skill.tools]


def instructions_for_tools(tools):
    """Load only prompts whose tools were explicitly selected."""
    from .skill_loader import get_skill_loader
    from src.agent.routing.policy import ALLOWED_SKILLS
    names = {tool.name for tool in tools}
    loader = get_skill_loader()
    # Metadata is read without importing or calling unrelated tool modules.
    import frontmatter
    prompts = []
    for skill_name in sorted(ALLOWED_SKILLS):
        metadata, prompt = frontmatter.parse((loader.skills_dir / skill_name / "Skill.md").read_text())
        declared = {name for group in metadata.get("tools", []) for name in group.get("names", [])}
        if declared.intersection(names):
            prompts.append(prompt)
    return "\n\n".join(prompts)
