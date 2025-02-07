def aggregate_expert_opinions(expert_opinions):
    """
    Aggregates expert opinions using a mathematical method.

    Args:
        expert_opinions (list): A list of expert opinions (numeric values).

    Returns:
        float: Aggregated risk score.
    """
    # Example: Simple average of expert opinions
    total_opinions = len(expert_opinions)
    total_score = sum(expert_opinions)
    aggregated_score = total_score / total_opinions
    return aggregated_score

def main():
    # Example expert opinions (replace with actual expert opinions)
    expert_opinions = [0.8, 0.9, 0.7, 0.85]

    # Aggregate expert opinions
    aggregated_risk_score = aggregate_expert_opinions(expert_opinions)

    print(f"Aggregated risk score: {aggregated_risk_score}")

if __name__ == "__main__":
    main()
