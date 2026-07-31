"""
Module 1: Ontology Exporter (JSON / OWL)
=========================================
Exports the validated ontology into standardized machine-readable
formats for Semantic Web integration and academic publication.

Export Formats:
1. JSON: Nested hierarchy with metadata, provenance, and examples
2. OWL/RDF: Web Ontology Language using rdflib for Semantic Web

The exports include:
- Taxonomic (is-a) relationships
- Mereological (part-of) relationships
- OntoClean metaproperty annotations
- Provenance metadata (source games, rule excerpts)

Responsibilities (per proposal):
- Export finalized ontology in JSON and OWL formats
- Define taxonomic and mereological relations explicitly
- Embed provenance metadata for academic reproducibility
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ONTOLOGY_OUTPUT_DIR
from shared import setup_logger, save_json

logger = setup_logger("OntologyExporter")

# ─── Namespace Definitions ───────────────────────────────────
ONTOLOGY_IRI = "http://example.org/boardgame-mechanics-ontology#"
ONTOLOGY_VERSION = "1.0.0"


# ─── JSON Export ─────────────────────────────────────────────
def export_to_json(
    nodes: Dict,
    labeled_clusters: Dict,
    taxonomy_tree: Dict = None,
    validation_report: Dict = None,
    output_path: str = None
) -> str:
    """
    Export the complete ontology as a structured JSON document.
    
    The JSON includes:
    - Ontology metadata and provenance
    - Hierarchical taxonomy structure
    - Individual mechanic nodes with MDA classifications
    - OntoClean metaproperty annotations
    - Cross-references to source games
    
    Returns:
        Path to the saved JSON file
    """
    if output_path is None:
        output_path = os.path.join(ONTOLOGY_OUTPUT_DIR, "ontology.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build the export structure
    ontology = {
        "@context": {
            "ontology": ONTOLOGY_IRI,
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
        },
        "metadata": {
            "title": "AI-Driven Board Game Mechanics Ontology",
            "version": ONTOLOGY_VERSION,
            "created": datetime.now().isoformat(),
            "description": (
                "A hierarchical taxonomy of board game mechanics generated "
                "through AI-driven ontology extraction using Sentence-BERT "
                "embeddings, Hierarchical Agglomerative Clustering, and "
                "MDA framework alignment."
            ),
            "methodology": {
                "embedding_model": "Sentence-BERT (all-MiniLM-L6-v2)",
                "clustering": "Hierarchical Agglomerative Clustering (Ward's linkage)",
                "dimensionality_reduction": "UMAP",
                "labeling": "LLM-assisted (Gemini) with keyword-based fallback",
                "validation": "OntoClean metaproperty constraints",
            },
            "frameworks": ["MDA (Mechanics, Dynamics, Aesthetics)", "LUDES Game Mechanics Ontology"],
            "license": "CC-BY-4.0",
        },
        "top_level_categories": {
            "Algorithm": {
                "description": "Processes and player interactions — operable rules",
                "subcategories": ["Action", "Ruleset", "Turn Structure", "Conflict Resolution"],
            },
            "Data Representation": {
                "description": "System data structures — game components and state",
                "subcategories": ["Component", "State", "Spatial"],
            },
            "Player Interaction": {
                "description": "Patterns of player-to-player interaction",
                "subcategories": ["Cooperative", "Competitive", "Negotiation"],
            },
        },
        "mechanic_clusters": {},
        "taxonomy_tree": taxonomy_tree,
        "validation_summary": None,
    }
    
    # Add labeled clusters
    for cluster_id, cluster_info in labeled_clusters.items():
        cid = str(cluster_id)
        ontology["mechanic_clusters"][cid] = {
            "label": cluster_info.get("label", f"Cluster_{cluster_id}"),
            "mda_category": cluster_info.get("mda_category", "Unknown"),
            "definition": cluster_info.get("definition", ""),
            "parent": cluster_info.get("parent", ""),
            "members": cluster_info.get("members", []),
            "relations": {
                "is_a": cluster_info.get("parent", ""),
                "part_of": [],  # Mereological relations
            },
            "ontoclean": {},
        }
    
    # Add OntoClean annotations from nodes
    for node_name, node in nodes.items():
        node_dict = node.to_dict() if hasattr(node, 'to_dict') else node
        # Find matching cluster
        for cid, cluster in ontology["mechanic_clusters"].items():
            if cluster["label"] == node_name:
                cluster["ontoclean"] = {
                    "rigidity": node_dict.get("rigidity", "~R"),
                    "identity": node_dict.get("identity", "-I"),
                    "unity": node_dict.get("unity", "-U"),
                    "dependence": node_dict.get("dependence", "-D"),
                }
    
    # Add validation summary
    if validation_report:
        ontology["validation_summary"] = {
            "total_violations": validation_report.get("total_violations", 0),
            "errors": validation_report.get("errors", 0),
            "warnings": validation_report.get("warnings", 0),
            "status": "VALID" if validation_report.get("errors", 0) == 0 else "HAS_ERRORS",
        }
    
    save_json(ontology, output_path)
    logger.info(f"JSON ontology exported to: {output_path}")
    return output_path


# ─── OWL/RDF Export ──────────────────────────────────────────
def export_to_owl(
    nodes: Dict,
    labeled_clusters: Dict,
    output_path: str = None
) -> str:
    """
    Export the ontology in OWL/RDF format using rdflib.
    
    Creates an OWL ontology with:
    - Class hierarchy using rdfs:subClassOf
    - Annotation properties for MDA categories
    - OntoClean metaproperty annotations
    - SKOS labels and definitions
    
    Falls back to manual XML if rdflib is unavailable.
    
    Returns:
        Path to the saved OWL file
    """
    if output_path is None:
        output_path = os.path.join(ONTOLOGY_OUTPUT_DIR, "ontology.owl")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        return _export_owl_rdflib(nodes, labeled_clusters, output_path)
    except ImportError:
        logger.info("rdflib not available — generating manual OWL/XML")
        return _export_owl_manual(nodes, labeled_clusters, output_path)


def _export_owl_rdflib(nodes, labeled_clusters, output_path):
    """Export OWL using rdflib library."""
    from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL
    from rdflib.namespace import SKOS, DCTERMS
    
    g = Graph()
    BGM = Namespace(ONTOLOGY_IRI)
    g.bind("bgm", BGM)
    g.bind("owl", OWL)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    
    # Ontology declaration
    ont_uri = URIRef(ONTOLOGY_IRI.rstrip("#"))
    g.add((ont_uri, RDF.type, OWL.Ontology))
    g.add((ont_uri, RDFS.label, Literal("Board Game Mechanics Ontology")))
    g.add((ont_uri, DCTERMS.created, Literal(datetime.now().isoformat())))
    
    # Custom annotation properties for OntoClean
    rigidity_prop = BGM["hasRigidity"]
    identity_prop = BGM["hasIdentity"]
    unity_prop = BGM["hasUnity"]
    dependence_prop = BGM["hasDependence"]
    mda_prop = BGM["mdaCategory"]
    
    for prop in [rigidity_prop, identity_prop, unity_prop, dependence_prop, mda_prop]:
        g.add((prop, RDF.type, OWL.AnnotationProperty))
    
    # Create top-level classes
    top_classes = {
        "Algorithm": "Processes and player interactions",
        "DataRepresentation": "System data structures",
        "PlayerInteraction": "Player-to-player interaction patterns",
    }
    
    for cls_name, cls_desc in top_classes.items():
        cls_uri = BGM[cls_name]
        g.add((cls_uri, RDF.type, OWL.Class))
        g.add((cls_uri, RDFS.label, Literal(cls_name)))
        g.add((cls_uri, SKOS.definition, Literal(cls_desc)))
    
    # Create cluster classes
    for cluster_id, cluster_info in labeled_clusters.items():
        label = cluster_info.get("label", f"Cluster_{cluster_id}")
        safe_name = label.replace(" ", "_").replace("/", "_")
        cls_uri = BGM[safe_name]
        
        g.add((cls_uri, RDF.type, OWL.Class))
        g.add((cls_uri, RDFS.label, Literal(label)))
        
        definition = cluster_info.get("definition", "")
        if definition:
            g.add((cls_uri, SKOS.definition, Literal(definition)))
        
        # MDA category annotation
        mda_cat = cluster_info.get("mda_category", "")
        if mda_cat:
            g.add((cls_uri, mda_prop, Literal(mda_cat)))
        
        # Parent class relationship
        parent = cluster_info.get("parent", "Algorithm")
        parent_safe = parent.replace(" ", "").replace("/", "_")
        if parent_safe in top_classes or any(parent_safe == k for k in top_classes):
            g.add((cls_uri, RDFS.subClassOf, BGM[parent_safe]))
        
        # Add members as individuals
        for member in cluster_info.get("members", [])[:15]:
            member_safe = member.replace(" ", "_").replace("/", "_").replace("'", "")
            member_uri = BGM[member_safe]
            g.add((member_uri, RDF.type, cls_uri))
            g.add((member_uri, RDFS.label, Literal(member)))
    
    # Add OntoClean annotations
    for node_name, node in nodes.items():
        node_dict = node.to_dict() if hasattr(node, 'to_dict') else node
        safe_name = node_name.replace(" ", "_").replace("/", "_")
        node_uri = BGM[safe_name]
        
        if node_dict.get("rigidity"):
            g.add((node_uri, rigidity_prop, Literal(node_dict["rigidity"])))
        if node_dict.get("identity"):
            g.add((node_uri, identity_prop, Literal(node_dict["identity"])))
        if node_dict.get("unity"):
            g.add((node_uri, unity_prop, Literal(node_dict["unity"])))
        if node_dict.get("dependence"):
            g.add((node_uri, dependence_prop, Literal(node_dict["dependence"])))
    
    # Serialize
    g.serialize(destination=output_path, format="xml")
    logger.info(f"OWL ontology exported to: {output_path}")
    return output_path


def _export_owl_manual(nodes, labeled_clusters, output_path):
    """Generate OWL/XML manually without rdflib."""
    
    xml_lines = [
        '<?xml version="1.0"?>',
        f'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
        f'         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
        f'         xmlns:owl="http://www.w3.org/2002/07/owl#"',
        f'         xmlns:bgm="{ONTOLOGY_IRI}">',
        f'',
        f'  <owl:Ontology rdf:about="{ONTOLOGY_IRI.rstrip("#")}">',
        f'    <rdfs:label>Board Game Mechanics Ontology</rdfs:label>',
        f'  </owl:Ontology>',
    ]
    
    # Top-level classes
    for cls in ["Algorithm", "DataRepresentation", "PlayerInteraction"]:
        xml_lines.append(f'')
        xml_lines.append(f'  <owl:Class rdf:about="{ONTOLOGY_IRI}{cls}">')
        xml_lines.append(f'    <rdfs:label>{cls}</rdfs:label>')
        xml_lines.append(f'  </owl:Class>')
    
    # Cluster classes
    for cluster_id, cluster_info in labeled_clusters.items():
        label = cluster_info.get("label", f"Cluster_{cluster_id}")
        safe_name = label.replace(" ", "_").replace("/", "_")
        parent = cluster_info.get("parent", "Algorithm").replace(" ", "")
        definition = cluster_info.get("definition", "").replace("&", "&amp;").replace("<", "&lt;")
        
        xml_lines.append(f'')
        xml_lines.append(f'  <owl:Class rdf:about="{ONTOLOGY_IRI}{safe_name}">')
        xml_lines.append(f'    <rdfs:label>{label}</rdfs:label>')
        xml_lines.append(f'    <rdfs:subClassOf rdf:resource="{ONTOLOGY_IRI}{parent}"/>')
        if definition:
            xml_lines.append(f'    <rdfs:comment>{definition}</rdfs:comment>')
        xml_lines.append(f'  </owl:Class>')
    
    xml_lines.append(f'')
    xml_lines.append(f'</rdf:RDF>')
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_lines))
    
    logger.info(f"OWL ontology (manual XML) exported to: {output_path}")
    return output_path


# ─── Main Export Pipeline ────────────────────────────────────
def run_export_pipeline(
    nodes: Dict,
    labeled_clusters: Dict,
    taxonomy_tree: Dict = None,
    validation_report: Dict = None
) -> Dict:
    """
    Execute the full export pipeline, generating both JSON and OWL outputs.
    
    Returns:
        Dict with paths to exported files
    """
    json_path = export_to_json(
        nodes, labeled_clusters, taxonomy_tree, validation_report
    )
    
    owl_path = export_to_owl(nodes, labeled_clusters)
    
    return {
        "json_path": json_path,
        "owl_path": owl_path,
        "formats": ["JSON", "OWL/RDF"],
    }
