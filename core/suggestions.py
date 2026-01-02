def generate_suggestions(missing_keywords):
    suggestions = []
    for skill in list(missing_keywords)[:10]:
        suggestions.append(f"Add or highlight experience related to '{skill}'.")
    return suggestions
