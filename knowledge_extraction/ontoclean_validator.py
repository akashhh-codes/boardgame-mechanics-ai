"""
Module 1: OntoClean Metaproperty Validator
===========================================
Implements the OntoClean methodology for validating the ontological
adequacy and logical consistency of the extracted taxonomy.

OntoClean Metaproperties (Guarino & Welty):
1. Identity (+I/-I): Criteria for uniquely identifying instances
2. Rigidity (+R/-R/~R): Whether class membership is essential/constant
3. Unity (+U/-U/~U): Whether instances are integral wholes
4. Dependence (+D/-D): Whether existence depends on another entity

Validation Constraints:
- A rigid class cannot be subsumed under an anti-rigid class
- If parent carries +I, all subclasses must carry compatible identity
- Anti-rigid subclass cannot subsume a rigid class
- Unity constraints must be consistent along is-a paths

Responsibilities (per proposal):
- Apply OntoClean constraints to validate taxonomy
- Flag logical inconsistencies in parent-child relationships
- Generate validation reports for iterative improvement
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field, asdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ONTOLOGY_OUTPUT_DIR
from shared import setup_logger, save_json

logger = setup_logger("OntoClean")


# ─── Metaproperty Definitions ────────────────────────────────
class Rigidity(str, Enum):
    RIGID = "+R"        # Membership is essential (e.g., "Token" is always a token)
    ANTI_RIGID = "-R"   # Membership can change (e.g., "Current Player")
    NON_RIGID = "~R"    # Neither necessarily rigid nor anti-rigid


class Identity(str, Enum):
    CARRIES = "+I"      # Has an identity criterion
    NO_IDENTITY = "-I"  # No specific identity criterion
    INHERITS = "~I"     # Inherits identity from parent


class Unity(str, Enum):
    UNIFIED = "+U"      # Instances are integral wholes
    NO_UNITY = "-U"     # Instances may not be integral wholes
    INHERITS = "~U"     # Inherits unity from parent


class Dependence(str, Enum):
    DEPENDENT = "+D"    # Existence depends on another entity
    INDEPENDENT = "-D"  # Exists independently


# ─── Ontology Node with Metaproperties ───────────────────────
@dataclass
class OntologyNode:
    """Represents a node in the ontology with OntoClean metaproperties."""
    name: str
    node_type: str                          # "class" or "instance"
    rigidity: Rigidity = Rigidity.NON_RIGID
    identity: Identity = Identity.NO_IDENTITY
    unity: Unity = Unity.NO_UNITY
    dependence: Dependence = Dependence.INDEPENDENT
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    mda_category: str = ""
    definition: str = ""

    def to_dict(self):
        return {
            "name": self.name,
            "node_type": self.node_type,
            "rigidity": self.rigidity.value,
            "identity": self.identity.value,
            "unity": self.unity.value,
            "dependence": self.dependence.value,
            "parent": self.parent,
            "children": self.children,
            "mda_category": self.mda_category,
            "definition": self.definition,
        }


# ─── Automatic Metaproperty Assignment ──────────────────────
def assign_metaproperties(
    label: str,
    mda_category: str,
    members: List[str]
) -> Dict:
    """
    Automatically assign OntoClean metaproperties based on
    MDA category and content analysis.
    
    Heuristic rules:
    - Data Representation/Components → Rigid (+R), has Identity (+I)
    - Algorithm/Actions → Anti-Rigid (-R), context-dependent
    - Algorithm/Rulesets → Non-Rigid (~R), systemic
    - Player Interactions → Anti-Rigid (-R), dependent (+D)
    """
    label_lower = label.lower()
    category_lower = mda_category.lower()
    
    # Default metaproperties
    rigidity = Rigidity.NON_RIGID
    identity = Identity.NO_IDENTITY
    unity = Unity.NO_UNITY
    dependence = Dependence.INDEPENDENT
    
    # Data Representation — physical components are rigid
    if "data representation" in category_lower or "component" in category_lower:
        rigidity = Rigidity.RIGID
        identity = Identity.CARRIES
        unity = Unity.UNIFIED
        dependence = Dependence.INDEPENDENT
    
    # Spatial representations — rigid, unified
    elif "spatial" in category_lower:
        rigidity = Rigidity.RIGID
        identity = Identity.CARRIES
        unity = Unity.UNIFIED
        dependence = Dependence.INDEPENDENT
    
    # State trackers — depend on game context
    elif "state" in category_lower:
        rigidity = Rigidity.NON_RIGID
        identity = Identity.CARRIES
        unity = Unity.UNIFIED
        dependence = Dependence.DEPENDENT
    
    # Algorithm/Action — anti-rigid (can change based on context)
    elif "action" in category_lower:
        rigidity = Rigidity.ANTI_RIGID
        identity = Identity.CARRIES
        unity = Unity.UNIFIED
        dependence = Dependence.DEPENDENT
    
    # Algorithm/Ruleset — non-rigid systemic rules
    elif "ruleset" in category_lower:
        rigidity = Rigidity.NON_RIGID
        identity = Identity.NO_IDENTITY
        unity = Unity.NO_UNITY
        dependence = Dependence.INDEPENDENT
    
    # Player Interaction — anti-rigid, dependent on players
    elif "player" in category_lower or "interaction" in category_lower:
        rigidity = Rigidity.ANTI_RIGID
        identity = Identity.NO_IDENTITY
        unity = Unity.NO_UNITY
        dependence = Dependence.DEPENDENT
    
    # Turn Structure — non-rigid, dependent on game state
    elif "turn" in category_lower or "structure" in category_lower:
        rigidity = Rigidity.NON_RIGID
        identity = Identity.CARRIES
        unity = Unity.UNIFIED
        dependence = Dependence.DEPENDENT
    
    return {
        "rigidity": rigidity,
        "identity": identity,
        "unity": unity,
        "dependence": dependence,
    }


# ─── OntoClean Constraint Validation ────────────────────────
@dataclass
class ValidationViolation:
    """Represents an OntoClean constraint violation."""
    rule: str
    parent_node: str
    child_node: str
    parent_property: str
    child_property: str
    severity: str  # "ERROR" or "WARNING"
    explanation: str

    def to_dict(self):
        return {
            "rule": self.rule,
            "parent_node": self.parent_node,
            "child_node": self.child_node,
            "parent_property": self.parent_property,
            "child_property": self.child_property,
            "severity": self.severity,
            "explanation": self.explanation,
        }


def validate_ontology(nodes: Dict[str, OntologyNode]) -> List[ValidationViolation]:
    """
    Validate the entire ontology against OntoClean constraints.
    
    Constraint Rules:
    1. RIGIDITY: A rigid class (+R) cannot be subsumed under an 
       anti-rigid class (-R)
    2. IDENTITY: If a parent carries an identity criterion (+I),
       subclasses must carry a compatible identity criterion
    3. UNITY: If a parent has unity (+U), subclasses must maintain
       the same unity criterion
    4. ANTI-RIGID SUBSUMPTION: An anti-rigid class cannot have
       a rigid subclass
    
    Args:
        nodes: Dict mapping node names to OntologyNode objects
        
    Returns:
        List of ValidationViolation objects for any constraint breaches
    """
    violations = []
    
    for node_name, node in nodes.items():
        if node.parent is None:
            continue
        
        parent = nodes.get(node.parent)
        if parent is None:
            continue
        
        # Rule 1: Rigid cannot be subsumed under Anti-Rigid
        if (node.rigidity == Rigidity.RIGID and 
            parent.rigidity == Rigidity.ANTI_RIGID):
            violations.append(ValidationViolation(
                rule="Rigidity Constraint",
                parent_node=parent.name,
                child_node=node.name,
                parent_property=parent.rigidity.value,
                child_property=node.rigidity.value,
                severity="ERROR",
                explanation=(
                    f"Rigid class '{node.name}' cannot be subsumed under "
                    f"anti-rigid class '{parent.name}'. A rigid entity's "
                    f"membership is essential, but an anti-rigid parent "
                    f"implies membership can change."
                )
            ))
        
        # Rule 2: Identity inheritance constraint
        if (parent.identity == Identity.CARRIES and 
            node.identity == Identity.NO_IDENTITY):
            violations.append(ValidationViolation(
                rule="Identity Constraint",
                parent_node=parent.name,
                child_node=node.name,
                parent_property=parent.identity.value,
                child_property=node.identity.value,
                severity="WARNING",
                explanation=(
                    f"Parent '{parent.name}' carries identity criterion (+I), "
                    f"but child '{node.name}' has no identity criterion (-I). "
                    f"Subclasses should inherit or define compatible identity."
                )
            ))
        
        # Rule 3: Unity consistency
        if (parent.unity == Unity.UNIFIED and 
            node.unity == Unity.NO_UNITY):
            violations.append(ValidationViolation(
                rule="Unity Constraint",
                parent_node=parent.name,
                child_node=node.name,
                parent_property=parent.unity.value,
                child_property=node.unity.value,
                severity="WARNING",
                explanation=(
                    f"Parent '{parent.name}' has unity (+U), but child "
                    f"'{node.name}' lacks unity (-U). Unity should be "
                    f"consistent along is-a paths."
                )
            ))
        
        # Rule 4: Anti-rigid parent cannot have rigid children
        if (parent.rigidity == Rigidity.ANTI_RIGID and 
            node.rigidity == Rigidity.RIGID):
            violations.append(ValidationViolation(
                rule="Anti-Rigid Subsumption",
                parent_node=parent.name,
                child_node=node.name,
                parent_property=parent.rigidity.value,
                child_property=node.rigidity.value,
                severity="ERROR",
                explanation=(
                    f"Anti-rigid parent '{parent.name}' cannot have "
                    f"rigid child '{node.name}'. This violates the "
                    f"fundamental OntoClean subsumption constraint."
                )
            ))
    
    # Summary
    errors = sum(1 for v in violations if v.severity == "ERROR")
    warnings = sum(1 for v in violations if v.severity == "WARNING")
    logger.info(f"OntoClean validation: {errors} errors, {warnings} warnings")
    
    return violations


# ─── Build & Validate from Labeled Clusters ─────────────────
def build_and_validate_ontology(
    labeled_clusters: Dict[int, Dict]
) -> Tuple[Dict[str, OntologyNode], List[ValidationViolation]]:
    """
    Build ontology nodes from labeled clusters and run OntoClean validation.
    
    Creates a two-level hierarchy:
    - Top level: MDA categories (Algorithm, Data Representation, Player Interaction)
    - Second level: Cluster labels
    - Leaf level: Individual mechanics
    
    Returns:
        Tuple of (ontology_nodes_dict, violations_list)
    """
    nodes = {}
    
    # Create top-level MDA category nodes
    for category_name, category_info in {
        "Algorithm": {"rigidity": Rigidity.NON_RIGID, "identity": Identity.CARRIES, "unity": Unity.UNIFIED, "dependence": Dependence.INDEPENDENT},
        "Data Representation": {"rigidity": Rigidity.RIGID, "identity": Identity.CARRIES, "unity": Unity.UNIFIED, "dependence": Dependence.INDEPENDENT},
        "Player Interaction": {"rigidity": Rigidity.ANTI_RIGID, "identity": Identity.NO_IDENTITY, "unity": Unity.NO_UNITY, "dependence": Dependence.DEPENDENT},
    }.items():
        nodes[category_name] = OntologyNode(
            name=category_name,
            node_type="class",
            parent=None,
            **category_info
        )
    
    # Create cluster-level nodes
    for cluster_id, cluster_info in labeled_clusters.items():
        label = cluster_info.get("label", f"Cluster_{cluster_id}")
        mda_cat = cluster_info.get("mda_category", "Algorithm/Action")
        members = cluster_info.get("members", [])
        definition = cluster_info.get("definition", "")
        
        # Determine parent from MDA category
        parent_name = mda_cat.split("/")[0] if "/" in mda_cat else "Algorithm"
        
        # Assign metaproperties
        metaprops = assign_metaproperties(label, mda_cat, members)
        
        nodes[label] = OntologyNode(
            name=label,
            node_type="class",
            parent=parent_name,
            children=members[:20],  # Cap for readability
            mda_category=mda_cat,
            definition=definition,
            **metaprops
        )
        
        # Add to parent's children
        if parent_name in nodes:
            nodes[parent_name].children.append(label)
    
    # Validate
    violations = validate_ontology(nodes)
    
    # Save validation report
    report = {
        "total_nodes": len(nodes),
        "total_violations": len(violations),
        "errors": sum(1 for v in violations if v.severity == "ERROR"),
        "warnings": sum(1 for v in violations if v.severity == "WARNING"),
        "violations": [v.to_dict() for v in violations],
        "nodes": {name: node.to_dict() for name, node in nodes.items()},
    }
    
    output_path = os.path.join(ONTOLOGY_OUTPUT_DIR, "ontoclean_report.json")
    save_json(report, output_path)
    logger.info(f"OntoClean report saved to: {output_path}")
    
    return nodes, violations


# ─── CLI Entry Point ──────────────────────────────────────────
if __name__ == "__main__":
    # Test with sample data
    sample_clusters = {
        0: {"label": "Resource Management", "mda_category": "Algorithm/Action", "members": ["Trading", "Collecting"]},
        1: {"label": "Physical Components", "mda_category": "Data Representation/Component", "members": ["Cards", "Dice"]},
    }
    nodes, violations = build_and_validate_ontology(sample_clusters)
    for v in violations:
        print(f"[{v.severity}] {v.rule}: {v.explanation}")
