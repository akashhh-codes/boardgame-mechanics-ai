def get_cluster_profiles():
    """Returns static rule paradigms mapped out by the K-Means analysis."""
    return {
        0: {"name": "Card-Driven Tactical Hand Management", "desc": "Rules focus heavy emphasis on playing hand combinations, scaling player powers over time, and resolving actions using dice metrics."},
        1: {"name": "Heavy Political & Territory Strategy", "desc": "Rules dictate map-control boundaries, military or economic area majority, and long-term civilization resource engine management."},
        2: {"name": "Engine Building & Tactical Card Play", "desc": "Rules emphasize card synergy, private board expansion, set-collection optimization, and high-efficiency engine loops."},
        3: {"name": "Drafting & Closed-Economy Optimization", "desc": "Rules center on tight drafting pools, direct resource conversion, building placement, and contract fulfillment constraints."},
        4: {"name": "Heavy Macro-Economic Logistics", "desc": "Complex financial management simulation rules. Features network construction, supply-demand curves, and worker optimization layouts."},
        5: {"name": "Narrative Campaign & Cooperative Survival", "desc": "Rules enforce complete player cooperation against automated game AI, scenario-driven legacy maps, and custom player asymmetric roles."},
        6: {"name": "Spatial Optimization & Worker Placement", "desc": "Rules dictate alternate engine building using spatial tile placement layouts, shared worker pools, and drafting resource spaces."},
        7: {"name": "Grand Asymmetric Conflicts & Dice Resolution", "desc": "High-stakes grand strategy rules featuring asymmetrical starting conditions, hidden deployment, and tactical dice combat grids."}
    }
