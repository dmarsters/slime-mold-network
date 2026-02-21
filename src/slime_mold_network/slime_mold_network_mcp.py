"""
Slime Mold Network MCP Server

Morphological computation aesthetics based on Physarum polycephalum growth dynamics.
Empirically grounded in Bajpai et al. (2025) morphological data.

Layer 1: Pure taxonomy lookup (0 tokens)
Layer 2: Deterministic mappings from empirical data (0 tokens)
Layer 3: Synthesis preparation for Claude

Research Attribution:
- Michael Levin Lab (Tufts University) - Morphological computation
- Philip Kurian Lab (Howard University) - Quantum biology bounds
- Toshiyuki Nakagaki - Physarum problem-solving
"""

from fastmcp import FastMCP
import math
from typing import Dict, Any, List, Optional

mcp = FastMCP("slime-mold-network")

# ============================================================================
# LAYER 1: TAXONOMY (Pure Lookup - 0 tokens)
# ============================================================================

GROWTH_PHASES = {
    "acclimation": {
        "name": "Acclimation Phase",
        "description": "Initial adaptation to environment, circular seed establishing contact",
        "circularity_range": (0.75, 1.0),
        "fractal_dimension_range": (1.70, 1.75),
        "typical_duration_hours": (0, 2),
        "visual_character": "Smooth circular boundary, minimal protrusions, establishing metabolic gradient",
        "computational_activity": "Environmental sensing, initial ATP distribution setup",
        "categorical_family": "morphisms"
    },
    "protuberant_exploration": {
        "name": "Protuberant Exploration",
        "description": "Asymmetric pseudopod extension seeking nutrients or space",
        "circularity_range": (0.2, 0.5),
        "fractal_dimension_range": (1.80, 1.90),
        "typical_duration_hours": (3, 12),
        "visual_character": "Irregular dendritic sprawl, multiple pseudopod-like extensions ~0.45mm width",
        "computational_activity": "Active pathfinding, TSP optimization, high fractal complexity",
        "categorical_family": "morphisms"
    },
    "radial_expansion": {
        "name": "Radial Expansion",
        "description": "Symmetric equiradial growth, higher circularity maintenance",
        "circularity_range": (0.5, 0.8),
        "fractal_dimension_range": (1.72, 1.80),
        "typical_duration_hours": (2, 20),
        "visual_character": "Relatively uniform circular expansion, smooth edges, balanced pseudopods",
        "computational_activity": "Distributed computation, metabolic homeostasis",
        "categorical_family": "morphisms"
    },
    "network_formation": {
        "name": "Network Formation",
        "description": "Emergence of vein lattice structure, optimized transport network",
        "circularity_range": (0.15, 0.4),
        "fractal_dimension_range": (1.78, 1.92),
        "typical_duration_hours": (8, 24),
        "visual_character": "Visible vein network, hierarchical branching, shuttle-streaming cytoplasm",
        "computational_activity": "Graph optimization, minimum-risk pathfinding, network consolidation",
        "categorical_family": "objects"
    },
    "boundary_adaptation": {
        "name": "Boundary Adaptation",
        "description": "Edge detection and constraint response, approaching NESS transition",
        "circularity_range": (0.05, 0.25),
        "fractal_dimension_range": (1.70, 1.85),
        "typical_duration_hours": (10, 30),
        "visual_character": "Complex edge topology, arena-constrained morphology, peripheral nuclei concentration",
        "computational_activity": "Boundary sensing, spatial constraint integration",
        "categorical_family": "constraints"
    },
    "steady_state": {
        "name": "Non-Equilibrium Steady State (NESS)",
        "description": "Stabilized area/perimeter, minimal morphological change, linear computational scaling",
        "circularity_range": (0.05, 0.20),
        "fractal_dimension_range": (1.72, 1.92),
        "typical_duration_hours": (12, 72),
        "visual_character": "Stable boundary complexity, consolidated vein network, plateau morphology",
        "computational_activity": "Linear scaling regime, metabolic equilibrium with environment",
        "categorical_family": "constraints"
    }
}

NETWORK_TOPOLOGY = {
    "connected_lattice": {
        "name": "Connected Vein Lattice",
        "description": "Continuous network with intact cytoplasmic streaming paths",
        "connectivity": 0.95,
        "vein_density": "high",
        "visual_signature": "Tree-like hierarchical branching, clear primary/secondary veins",
        "transport_efficiency": 0.92,
        "computational_capacity": "Maximum - all oscillators coupled",
        "typical_strain": "Vein network-connected Japanese",
        "categorical_family": "objects"
    },
    "disrupted_lawn": {
        "name": "Disrupted Lawn Network",
        "description": "Fragmented network from mechanical disruption, independent regions",
        "connectivity": 0.35,
        "vein_density": "low",
        "visual_signature": "Patchy regions with weak interconnection, semi-autonomous domains",
        "transport_efficiency": 0.45,
        "computational_capacity": "Reduced - weakly coupled or isolated oscillators",
        "typical_strain": "Vein network-disrupted Japanese (scraped)",
        "categorical_family": "objects"
    },
    "minimal_veins": {
        "name": "Minimal Vein Structure",
        "description": "Early stage before prominent vein emergence",
        "connectivity": 0.60,
        "vein_density": "minimal",
        "visual_signature": "Amorphous protoplasmic mass, nascent flow channels",
        "transport_efficiency": 0.55,
        "computational_capacity": "Developing - local oscillations forming",
        "typical_strain": "Early growth phase all strains",
        "categorical_family": "morphisms"
    },
    "dense_mesh": {
        "name": "Dense Mesh Network",
        "description": "Highly reticulated mature network with redundant paths",
        "connectivity": 0.98,
        "vein_density": "very_high",
        "visual_signature": "Complex anastomosing network, multiple parallel paths",
        "transport_efficiency": 0.97,
        "computational_capacity": "Maximum - highly coupled multi-oscillator system",
        "typical_strain": "Old Japanese, mature Carolina",
        "categorical_family": "objects"
    }
}

METABOLIC_STATE = {
    "starved_aggressive": {
        "name": "Starved Aggressive Explorer",
        "description": "Nutrient-deprived state driving rapid extensive exploration",
        "atp_periphery_concentration_mM": 2.0,
        "atp_center_concentration_mM": 0.3,
        "growth_rate_multiplier": 1.45,
        "exploration_bias": 0.92,
        "area_at_24h_cm2": 15.5,
        "perimeter_at_24h_cm": 85.0,
        "chemical_ops_24h": 1.29e36,
        "visual_character": "Extensive dendritic sprawl, aggressive pseudopod projection",
        "categorical_family": "morphisms"
    },
    "well_fed_conservative": {
        "name": "Well-Fed Conservative Grower",
        "description": "Adequate nutrients allowing measured radial expansion",
        "atp_periphery_concentration_mM": 2.0,
        "atp_center_concentration_mM": 0.9,
        "growth_rate_multiplier": 1.0,
        "exploration_bias": 0.55,
        "area_at_24h_cm2": 11.0,
        "perimeter_at_24h_cm": 60.0,
        "chemical_ops_24h": 9.02e35,
        "visual_character": "Balanced growth, moderate protuberances, controlled expansion",
        "categorical_family": "morphisms"
    },
    "mature_stable": {
        "name": "Mature Stable (NESS)",
        "description": "Post-NESS metabolic equilibrium with environment",
        "atp_periphery_concentration_mM": 2.0,
        "atp_center_concentration_mM": 0.85,
        "growth_rate_multiplier": 0.15,
        "exploration_bias": 0.10,
        "area_at_24h_cm2": 10.5,
        "perimeter_at_24h_cm": 40.0,
        "chemical_ops_24h": 3.86e35,
        "visual_character": "Minimal boundary change, consolidated network, stable morphology",
        "categorical_family": "constraints"
    }
}

STRAIN_PROFILES = {
    "young_japanese": {
        "name": "Young Japanese (Sonobe)",
        "age_days_since_sclerotia": 27,
        "ness_transition_hours": 11.17,
        "max_fractal_dimension": 1.83,
        "fractal_peak_time_hours": 7.5,
        "circularity_decay_rate": 0.185,
        "growth_character": "Fast protuberant initial growth, early boundary touch",
        "computational_bounds_24h": {
            "chemical_atp": 9.02e35,
            "kinetic_energy": 1.03e23,
            "quantum_optical": 1.48e22,
            "hydrodynamic": 1.66e6
        },
        "allometric_exponent": 1.13,
        "categorical_family": "objects"
    },
    "old_japanese": {
        "name": "Old Japanese (Sonobe)",
        "age_days_since_sclerotia": 49,
        "ness_transition_hours": 51.43,
        "max_fractal_dimension": 1.92,
        "fractal_peak_time_hours": 48.0,
        "circularity_decay_rate": 0.082,
        "growth_character": "Slow equiradial start, eventually highest fractal complexity",
        "computational_bounds_24h": {
            "chemical_atp": 5.97e35,
            "kinetic_energy": 6.09e21,
            "quantum_optical": 9.77e21,
            "hydrodynamic": 3.18e5
        },
        "allometric_exponent": 1.52,
        "categorical_family": "objects"
    },
    "carolina": {
        "name": "Carolina Biological",
        "age_days_since_sclerotia": 29,
        "ness_transition_hours": 12.63,
        "max_fractal_dimension": 1.72,
        "fractal_peak_time_hours": 48.0,
        "circularity_decay_rate": 0.145,
        "growth_character": "Slowest expansion, bi-sigmoid growth pattern, lowest fractal peak",
        "computational_bounds_24h": {
            "chemical_atp": 3.86e35,
            "kinetic_energy": 7.30e21,
            "quantum_optical": 6.31e21,
            "hydrodynamic": 6.77e5
        },
        "allometric_exponent": 0.89,
        "categorical_family": "objects"
    },
    "young_japanese_starved": {
        "name": "Young Japanese Starved",
        "age_days_since_sclerotia": 27,
        "ness_transition_hours": 13.19,
        "max_fractal_dimension": 1.85,
        "fractal_peak_time_hours": 8.0,
        "circularity_decay_rate": 0.195,
        "growth_character": "Most aggressive exploration, highest area coverage, nutrient-seeking",
        "computational_bounds_24h": {
            "chemical_atp": 1.29e36,
            "kinetic_energy": 4.65e22,
            "quantum_optical": 2.11e22,
            "hydrodynamic": 1.28e6
        },
        "allometric_exponent": 1.10,
        "categorical_family": "objects"
    }
}

BOUNDARY_CHARACTER = {
    "smooth_circular": {
        "name": "Smooth Circular",
        "description": "Early growth minimal irregularity",
        "circularity": 0.90,
        "fractal_dimension": 1.72,
        "pseudopod_count": 0,
        "edge_complexity": "minimal",
        "visual_signature": "Clean curves, few inflections, geometric simplicity",
        "categorical_family": "constraints"
    },
    "moderate_irregularity": {
        "name": "Moderate Irregularity",
        "description": "Intermediate protrusions forming",
        "circularity": 0.45,
        "fractal_dimension": 1.78,
        "pseudopod_count": 8,
        "edge_complexity": "moderate",
        "visual_signature": "Multiple lobes, ~0.45mm pseudopod width, asymmetric",
        "categorical_family": "morphisms"
    },
    "highly_dendritic": {
        "name": "Highly Dendritic",
        "description": "Many pseudopods, low circularity",
        "circularity": 0.18,
        "fractal_dimension": 1.85,
        "pseudopod_count": 18,
        "edge_complexity": "high",
        "visual_signature": "Tree-like branching, numerous finger-like projections",
        "categorical_family": "morphisms"
    },
    "fractal_edge": {
        "name": "Fractal Edge",
        "description": "Peak complexity, maximum fractal dimension",
        "circularity": 0.08,
        "fractal_dimension": 1.92,
        "pseudopod_count": 25,
        "edge_complexity": "maximum",
        "visual_signature": "Self-similar boundary detail across scales, space-filling tendency",
        "categorical_family": "constraints"
    }
}

SCALE_CONTEXT = {
    "microscopic_detail": {
        "name": "Microscopic Detail",
        "scale": "10-100 micrometers",
        "visible_features": "Individual cilia, nuclear distribution, actin fiber bundles, ATP gradient",
        "imaging_modality": "Fluorescence microscopy (DAPI for nuclei, phalloidin for actin)",
        "visual_vocabulary": "Punctate nuclei at periphery, fiber network texture, luminous metabolic gradient",
        "research_context": "Quantum optical bounds (~10^22 ops), superradiant states in actin"
    },
    "mesoscopic_network": {
        "name": "Mesoscopic Network",
        "scale": "0.1-1 cm",
        "visible_features": "Vein structure, pseudopod boundaries, cytoplasmic streaming patterns",
        "imaging_modality": "High-resolution scanner (1600 dpi), stereomicroscopy",
        "visual_vocabulary": "Branching veins ~0.45mm width, oscillating protoplasm, network topology",
        "research_context": "Hydrodynamic bounds (~10^6 ops), 60-140s oscillation period"
    },
    "macroscopic_body": {
        "name": "Macroscopic Body",
        "scale": "1-20 cm",
        "visible_features": "Overall morphology, growth phase, boundary character, circularity",
        "imaging_modality": "Flatbed scanning, timelapse photography",
        "visual_vocabulary": "Macroplasmodial form, perimeter complexity, fractal dimension",
        "research_context": "Kinetic bounds (~10^23 ops), chemical ATP bounds (~10^36 ops)"
    }
}

# ============================================================================
# LAYER 2: DETERMINISTIC MAPPINGS (0 tokens)
# ============================================================================

def compute_circularity(area: float, perimeter: float) -> float:
    """
    Deterministic circularity calculation.
    
    Circularity = 4π × Area / Perimeter²
    
    Returns:
        1.0 = perfect circle
        <1.0 = irregular/protuberant morphology
    """
    if perimeter == 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter ** 2)

def sigmoid_growth_model(t: float, alpha: float, beta: float, gamma: float) -> float:
    """
    Sigmoid growth model from paper Eq. 26.
    
    A(t) = α / (1 + e^(-β(t - γ)))
    
    Args:
        t: Time in hours
        alpha: Asymptotic maximum value
        beta: Growth rate
        gamma: Inflection point time
    """
    return alpha / (1 + math.exp(-beta * (t - gamma)))

def atp_distribution_sigmoid(r: float, r_max: float, rho_0: float = 61.2) -> float:
    """
    ATP concentration distribution from paper Eq. 11.
    
    ρ_E(r) = ρ_0 × (tanh(1.472 × r/r_max) + 0.1)
    
    Args:
        r: Radial distance from center (0 to r_max)
        r_max: Maximum radius (perimeter at time t)
        rho_0: Frontal ATP energy density (mJ/cm³)
    
    Returns:
        ATP energy density at radius r
    """
    normalized_r = r / r_max if r_max > 0 else 0
    return rho_0 * (math.tanh(1.472 * normalized_r) + 0.1)

def estimate_ness_transition(alpha: float, beta: float, gamma: float, epsilon: float = 0.15) -> float:
    """
    Estimate NESS transition time from sigmoid parameters.
    
    From paper: t_NESS is earliest time where dA/dt ≤ ε × (dA/dt)_max
    
    Args:
        alpha, beta, gamma: Sigmoid fit parameters
        epsilon: Threshold fraction (default 0.15 = 15%)
    
    Returns:
        Estimated NESS transition time in hours
    """
    # Maximum growth rate occurs at inflection point (gamma)
    max_rate = alpha * beta / 4  # Derivative of sigmoid at inflection
    threshold = epsilon * max_rate
    
    # Solve for t where derivative equals threshold
    # dA/dt = α·β·e^(-β(t-γ)) / (1 + e^(-β(t-γ)))²
    # Approximate: occurs when sigmoid is ~95% of asymptote
    t_ness = gamma + (1 / beta) * math.log(19)  # 95% = 20/(1+19) when exp(-β(t-γ))=1/19
    
    return t_ness

def map_time_to_growth_phase(t: float, ness_time: float) -> str:
    """
    Map elapsed time to growth phase (0 tokens).
    
    Args:
        t: Time in hours since start
        ness_time: NESS transition time for this strain
    
    Returns:
        Growth phase ID
    """
    if t < 2:
        return "acclimation"
    elif t < 0.5 * ness_time:
        return "protuberant_exploration"
    elif t < 0.8 * ness_time:
        return "network_formation"
    elif t < ness_time:
        return "boundary_adaptation"
    else:
        return "steady_state"

def map_circularity_to_boundary(circularity: float, fractal_dim: float) -> str:
    """
    Map morphological indices to boundary character (0 tokens).
    
    Args:
        circularity: 0.0-1.0
        fractal_dim: 1.7-1.92
    
    Returns:
        Boundary character ID
    """
    if circularity > 0.75:
        return "smooth_circular"
    elif circularity > 0.35:
        return "moderate_irregularity"
    elif fractal_dim > 1.88:
        return "fractal_edge"
    else:
        return "highly_dendritic"

def compute_computational_bounds(
    area_cm2: float,
    perimeter_cm: float,
    perimeter_rate_cm_per_h: float,
    strain_profile: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute computational capacity bounds from morphology (0 tokens).
    
    Based on paper Section 4 formulas.
    
    Args:
        area_cm2: Current area
        perimeter_cm: Current perimeter
        perimeter_rate_cm_per_h: dP/dt
        strain_profile: Strain parameters
    
    Returns:
        Dictionary of bounds in operations
    """
    # Constants from paper
    h_bar = 1.054571817e-34  # Planck constant / 2π (J·s)
    rho_0 = 61.2e-3  # ATP energy density (J/cm³)
    thickness = 100e-4  # 100 μm in cm
    n_actin = 2e8  # 2 fibers per 1000 μm² = 2×10^8 m^-2
    tau_superradiant = 1e-11  # 10 ps
    pseudopod_width = 0.045  # 0.45 mm in cm
    n_local_hydro = 0.017  # ops per second per pseudopod
    
    # Hydrodynamic bound (Eq. 10)
    hydro_bound = n_local_hydro * perimeter_cm / pseudopod_width
    
    # Chemical ATP bound (Eq. 13) - simplified integral result
    chem_bound = (0.664 * rho_0 * thickness * area_cm2) / (math.pi * h_bar)
    
    # Quantum optical bound (Eq. 23)
    qo_bound = (n_actin * area_cm2 * 1e-4) / tau_superradiant  # Convert cm² to m²
    
    # Kinetic energy bound - simplified (paper Eq. 18 complex integral)
    # Approximate: scales with area × velocity²
    rho_m = 1100  # kg/m³
    f_avg = 0.15  # Estimated frontal fraction
    ke_bound = (rho_m * f_avg * thickness * area_cm2 * 1e-4 * (perimeter_rate_cm_per_h / 3600)**2) / (2 * math.pi * h_bar)
    
    return {
        "hydrodynamic": hydro_bound,
        "chemical_atp": chem_bound,
        "quantum_optical": qo_bound,
        "kinetic_energy": ke_bound
    }

# ============================================================================
# LAYER 1 TOOLS: Taxonomy Lookup
# ============================================================================

@mcp.tool()
def list_growth_phases() -> Dict[str, Any]:
    """
    List all six growth phases with empirical parameters.
    
    Layer 1: Pure taxonomy lookup (0 tokens)
    
    Returns:
        Dictionary of growth phase taxonomy
    """
    return {
        "growth_phases": GROWTH_PHASES,
        "count": len(GROWTH_PHASES),
        "source": "Bajpai et al. (2025) morphological time-series data"
    }

@mcp.tool()
def list_network_topologies() -> Dict[str, Any]:
    """
    List network topology categories.
    
    Layer 1: Pure taxonomy lookup (0 tokens)
    
    Returns:
        Network topology taxonomy
    """
    return {
        "topologies": NETWORK_TOPOLOGY,
        "count": len(NETWORK_TOPOLOGY),
        "source": "Vein network connectivity experiments (Section 6.1.3)"
    }

@mcp.tool()
def list_strain_profiles() -> Dict[str, Any]:
    """
    List all four strain profiles with empirical bounds.
    
    Layer 1: Pure taxonomy lookup (0 tokens)
    
    Returns:
        Strain profile taxonomy with computational bounds
    """
    return {
        "strains": STRAIN_PROFILES,
        "count": len(STRAIN_PROFILES),
        "key_findings": {
            "highest_computational_capacity": "young_japanese_starved",
            "highest_fractal_dimension": "old_japanese (d_f = 1.92)",
            "fastest_ness_transition": "young_japanese (11.17 hours)",
            "allometric_scaling": "ν correlates with fractal dimension"
        },
        "source": "Table S2, Figures 5-8"
    }

@mcp.tool()
def get_scale_contexts() -> Dict[str, Any]:
    """
    Get scale context taxonomy for visualization.
    
    Layer 1: Pure taxonomy lookup (0 tokens)
    
    Returns:
        Scale context specifications
    """
    return {
        "scales": SCALE_CONTEXT,
        "count": len(SCALE_CONTEXT),
        "note": "Different scales reveal different computational mechanisms"
    }

# ============================================================================
# LAYER 2 TOOLS: Deterministic Mappings
# ============================================================================

@mcp.tool()
def map_time_to_morphology(
    time_hours: float,
    strain_id: str,
    metabolic_state_id: str
) -> Dict[str, Any]:
    """
    Map elapsed time to expected morphological state (0 tokens).
    
    Layer 2: Deterministic mapping from empirical strain profiles.
    
    Args:
        time_hours: Time elapsed since experiment start
        strain_id: One of strain profile IDs
        metabolic_state_id: One of metabolic state IDs
    
    Returns:
        Predicted morphological parameters at time t
    """
    if strain_id not in STRAIN_PROFILES:
        return {"error": f"Unknown strain: {strain_id}"}
    if metabolic_state_id not in METABOLIC_STATE:
        return {"error": f"Unknown metabolic state: {metabolic_state_id}"}
    
    strain = STRAIN_PROFILES[strain_id]
    metabolic = METABOLIC_STATE[metabolic_state_id]
    
    # Determine growth phase
    ness_time = strain["ness_transition_hours"]
    growth_phase_id = map_time_to_growth_phase(time_hours, ness_time)
    growth_phase = GROWTH_PHASES[growth_phase_id]
    
    # Estimate morphological indices using simplified sigmoid
    # Use strain-specific parameters (would need full fit params for precision)
    progress = min(time_hours / ness_time, 1.0)
    
    # Circularity decays from ~0.9 to ~0.1
    circularity = 0.9 * (1 - progress * strain["circularity_decay_rate"] * 5)
    circularity = max(0.05, circularity)
    
    # Fractal dimension peaks then stabilizes
    peak_time = strain["fractal_peak_time_hours"]
    if time_hours < peak_time:
        fractal_dim = 1.70 + (strain["max_fractal_dimension"] - 1.70) * (time_hours / peak_time)
    else:
        fractal_dim = strain["max_fractal_dimension"]
    
    # Boundary character from morphology
    boundary_id = map_circularity_to_boundary(circularity, fractal_dim)
    
    return {
        "time_hours": time_hours,
        "strain": strain_id,
        "metabolic_state": metabolic_state_id,
        "growth_phase": {
            "id": growth_phase_id,
            "name": growth_phase["name"],
            "categorical_family": growth_phase["categorical_family"]
        },
        "morphological_indices": {
            "circularity": round(circularity, 3),
            "fractal_dimension": round(fractal_dim, 3),
            "estimated_area_cm2": metabolic["area_at_24h_cm2"] * progress,
            "estimated_perimeter_cm": metabolic["perimeter_at_24h_cm"] * progress
        },
        "boundary_character": {
            "id": boundary_id,
            "name": BOUNDARY_CHARACTER[boundary_id]["name"]
        },
        "ness_status": "reached" if time_hours >= ness_time else f"{round((time_hours/ness_time)*100)}% to NESS",
        "methodology": "deterministic_morphological_mapping",
        "llm_cost": "0 tokens"
    }

@mcp.tool()
def compute_atp_gradient(
    radius_cm: float,
    max_radius_cm: float
) -> Dict[str, Any]:
    """
    Compute ATP concentration at radial distance (0 tokens).
    
    Layer 2: Deterministic sigmoid from paper Eq. 11.
    
    Args:
        radius_cm: Distance from body center
        max_radius_cm: Current maximum radius (perimeter/2π for circular)
    
    Returns:
        ATP energy density and concentration
    """
    # ATP concentration parameters from paper
    rho_0 = 61.2  # mJ/cm³ for 2mM ATP at periphery
    atp_energy_density = atp_distribution_sigmoid(radius_cm, max_radius_cm, rho_0)
    
    # Convert to mM concentration (7.3 kcal/mol ATP = 0.317 eV/molecule)
    # 2mM at periphery corresponds to rho_0
    atp_concentration_mM = 2.0 * (atp_energy_density / rho_0)
    
    return {
        "radius_cm": radius_cm,
        "max_radius_cm": max_radius_cm,
        "normalized_radius": radius_cm / max_radius_cm if max_radius_cm > 0 else 0,
        "atp_energy_density_mJ_per_cm3": round(atp_energy_density, 2),
        "atp_concentration_mM": round(atp_concentration_mM, 3),
        "gradient_profile": "Sigmoid: high at periphery, decays toward center",
        "visual_mapping": {
            "luminosity": atp_concentration_mM / 2.0,  # Normalized 0-1
            "description": "Bright glow at edges, fading toward center"
        },
        "source": "Hirose et al. (1980), Ueda et al. (1987)"
    }

@mcp.tool()
def estimate_computational_capacity(
    area_cm2: float,
    perimeter_cm: float,
    time_hours: float,
    strain_id: str
) -> Dict[str, Any]:
    """
    Estimate computational bounds from morphology (0 tokens).
    
    Layer 2: Deterministic calculation from paper Section 4.
    
    Args:
        area_cm2: Current area
        perimeter_cm: Current perimeter
        time_hours: Time elapsed
        strain_id: Strain profile ID
    
    Returns:
        Four computational capacity bounds
    """
    if strain_id not in STRAIN_PROFILES:
        return {"error": f"Unknown strain: {strain_id}"}
    
    strain = STRAIN_PROFILES[strain_id]
    
    # Estimate perimeter rate (simplified - would need time derivative)
    ness_time = strain["ness_transition_hours"]
    if time_hours < ness_time:
        # Growth phase - estimate rate
        perimeter_rate = perimeter_cm / max(time_hours, 0.5)  # Avoid div by zero
    else:
        # NESS - minimal growth
        perimeter_rate = 0.1
    
    bounds = compute_computational_bounds(
        area_cm2, perimeter_cm, perimeter_rate, strain
    )
    
    return {
        "strain": strain_id,
        "time_hours": time_hours,
        "morphology": {
            "area_cm2": area_cm2,
            "perimeter_cm": perimeter_cm,
            "circularity": compute_circularity(area_cm2, perimeter_cm)
        },
        "computational_bounds_ops_per_second": {
            "hydrodynamic": f"{bounds['hydrodynamic']:.2e}",
            "chemical_atp": f"{bounds['chemical_atp']:.2e}",
            "quantum_optical": f"{bounds['quantum_optical']:.2e}",
            "kinetic_energy": f"{bounds['kinetic_energy']:.2e}"
        },
        "interpretation": {
            "dominant_mechanism": "chemical_atp" if bounds['chemical_atp'] > 1e30 else "kinetic_energy",
            "hierarchy": "Chemical ATP >> Quantum Optical ≈ Kinetic >> Hydrodynamic",
            "note": "Chemical bound is upper limit; most energy dissipates as heat"
        },
        "source": "Margolus-Levitin bounds, Equations 10, 13, 18, 23",
        "methodology": "deterministic_bound_calculation",
        "llm_cost": "0 tokens"
    }

# ============================================================================
# LAYER 3 TOOLS: Synthesis Preparation
# ============================================================================

@mcp.tool()
def generate_network_visualization_params(
    growth_phase_id: str,
    network_topology_id: str,
    metabolic_state_id: str,
    boundary_character_id: str,
    scale_context_id: str,
    strain_id: Optional[str] = "young_japanese"
) -> Dict[str, Any]:
    """
    Generate complete visual parameters for slime mold network synthesis.
    
    Layer 3: Assembles deterministic parameters for Claude synthesis.
    
    Args:
        growth_phase_id: Growth phase
        network_topology_id: Network topology
        metabolic_state_id: Metabolic state
        boundary_character_id: Boundary character
        scale_context_id: Scale context
        strain_id: Optional strain profile
    
    Returns:
        Complete visualization specification
    """
    # Validate all IDs
    if growth_phase_id not in GROWTH_PHASES:
        return {"error": f"Unknown growth phase: {growth_phase_id}"}
    if network_topology_id not in NETWORK_TOPOLOGY:
        return {"error": f"Unknown network topology: {network_topology_id}"}
    if metabolic_state_id not in METABOLIC_STATE:
        return {"error": f"Unknown metabolic state: {metabolic_state_id}"}
    if boundary_character_id not in BOUNDARY_CHARACTER:
        return {"error": f"Unknown boundary character: {boundary_character_id}"}
    if scale_context_id not in SCALE_CONTEXT:
        return {"error": f"Unknown scale context: {scale_context_id}"}
    if strain_id not in STRAIN_PROFILES:
        return {"error": f"Unknown strain: {strain_id}"}
    
    # Retrieve taxonomy entries
    growth_phase = GROWTH_PHASES[growth_phase_id]
    network = NETWORK_TOPOLOGY[network_topology_id]
    metabolic = METABOLIC_STATE[metabolic_state_id]
    boundary = BOUNDARY_CHARACTER[boundary_character_id]
    scale = SCALE_CONTEXT[scale_context_id]
    strain = STRAIN_PROFILES[strain_id]
    
    return {
        "domain": "slime_mold_network",
        "parameters": {
            "growth_phase": {
                "id": growth_phase_id,
                "name": growth_phase["name"],
                "visual_character": growth_phase["visual_character"],
                "circularity_range": growth_phase["circularity_range"],
                "fractal_dimension_range": growth_phase["fractal_dimension_range"],
                "categorical_family": growth_phase["categorical_family"]
            },
            "network_topology": {
                "id": network_topology_id,
                "name": network["name"],
                "visual_signature": network["visual_signature"],
                "connectivity": network["connectivity"],
                "vein_density": network["vein_density"],
                "categorical_family": network["categorical_family"]
            },
            "metabolic_state": {
                "id": metabolic_state_id,
                "name": metabolic["name"],
                "atp_gradient": {
                    "periphery_mM": metabolic["atp_periphery_concentration_mM"],
                    "center_mM": metabolic["atp_center_concentration_mM"],
                    "profile": "sigmoid_decay"
                },
                "growth_rate_multiplier": metabolic["growth_rate_multiplier"],
                "visual_character": metabolic["visual_character"],
                "categorical_family": metabolic["categorical_family"]
            },
            "boundary_character": {
                "id": boundary_character_id,
                "name": boundary["name"],
                "circularity": boundary["circularity"],
                "fractal_dimension": boundary["fractal_dimension"],
                "pseudopod_count": boundary["pseudopod_count"],
                "visual_signature": boundary["visual_signature"],
                "categorical_family": boundary["categorical_family"]
            },
            "scale_context": {
                "id": scale_context_id,
                "name": scale["name"],
                "scale": scale["scale"],
                "visible_features": scale["visible_features"],
                "visual_vocabulary": scale["visual_vocabulary"]
            },
            "strain_profile": {
                "id": strain_id,
                "name": strain["name"],
                "max_fractal_dimension": strain["max_fractal_dimension"],
                "computational_character": f"Allometric exponent: {strain['allometric_exponent']}"
            }
        },
        "visual_synthesis_guidance": {
            "overall_form": growth_phase["visual_character"],
            "network_structure": network["visual_signature"],
            "edge_treatment": boundary["visual_signature"],
            "metabolic_gradient": f"ATP glow: {metabolic['atp_periphery_concentration_mM']}mM periphery → {metabolic['atp_center_concentration_mM']}mM center",
            "scale_details": scale["visible_features"],
            "color_palette": "Fluorescent microscopy: blue (DAPI nuclei), magenta (phalloidin actin), yellow-green (ATP bioluminescence)",
            "computational_metaphor": f"Network optimizes {network['transport_efficiency']*100}% efficiently, performing ~10^{int(math.log10(metabolic['chemical_ops_24h']))} ops/day"
        },
        "research_attribution": {
            "morphological_data": "Bajpai, Lucas-DeMott, Murugan, Levin, Kurian (2025)",
            "problem_solving": "Nakagaki et al. - TSP optimization, maze solving",
            "quantum_bounds": "Kurian Lab - Superradiance in protein architectures",
            "institutions": "Tufts University (Levin Lab), Howard University (Kurian Lab)"
        },
        "methodology": "layer_3_synthesis_preparation",
        "ready_for_claude_synthesis": True
    }

@mcp.tool()
def generate_temporal_sequence(
    start_time_hours: float,
    end_time_hours: float,
    num_steps: int,
    strain_id: str,
    metabolic_state_id: str,
    scale_context_id: str
) -> Dict[str, Any]:
    """
    Generate temporal sequence of morphological states.
    
    Layer 3: Time-series visualization preparation.
    
    Args:
        start_time_hours: Sequence start time
        end_time_hours: Sequence end time
        num_steps: Number of frames
        strain_id: Strain profile
        metabolic_state_id: Metabolic state
        scale_context_id: Scale context
    
    Returns:
        Sequence of morphological states for animation
    """
    if strain_id not in STRAIN_PROFILES:
        return {"error": f"Unknown strain: {strain_id}"}
    if metabolic_state_id not in METABOLIC_STATE:
        return {"error": f"Unknown metabolic state: {metabolic_state_id}"}
    if scale_context_id not in SCALE_CONTEXT:
        return {"error": f"Unknown scale context: {scale_context_id}"}
    
    strain = STRAIN_PROFILES[strain_id]
    metabolic = METABOLIC_STATE[metabolic_state_id]
    scale = SCALE_CONTEXT[scale_context_id]
    
    sequence = []
    time_step = (end_time_hours - start_time_hours) / (num_steps - 1)
    
    for i in range(num_steps):
        t = start_time_hours + i * time_step
        
        # Map time to morphology
        morph_state = map_time_to_morphology(t, strain_id, metabolic_state_id)
        
        # Get boundary character for this state
        boundary_id = morph_state["boundary_character"]["id"]
        network_id = "connected_lattice" if t > 8 else "minimal_veins"
        
        sequence.append({
            "frame": i,
            "time_hours": round(t, 2),
            "growth_phase": morph_state["growth_phase"]["name"],
            "morphology": morph_state["morphological_indices"],
            "boundary_character": morph_state["boundary_character"]["name"],
            "network_state": NETWORK_TOPOLOGY[network_id]["visual_signature"],
            "ness_status": morph_state["ness_status"]
        })
    
    return {
        "domain": "slime_mold_network",
        "sequence_type": "temporal_morphological_evolution",
        "strain": strain["name"],
        "metabolic_state": metabolic["name"],
        "scale": scale["name"],
        "duration_hours": end_time_hours - start_time_hours,
        "num_frames": num_steps,
        "frames": sequence,
        "visualization_notes": {
            "circularity_trend": "Decays from ~0.9 to ~0.1 (circular → dendritic)",
            "fractal_dimension_trend": f"Peaks at {strain['fractal_peak_time_hours']}h, max {strain['max_fractal_dimension']}",
            "ness_transition": f"Occurs at {strain['ness_transition_hours']}h",
            "key_visual_change": "Network emerges ~8h, boundary complexity peaks before NESS"
        },
        "animation_guidance": {
            "early_phase": "Rapid circular expansion with emerging protrusions",
            "mid_phase": "Dendritic exploration, vein network becomes visible",
            "late_phase": "Boundary stabilizes, network consolidates, minimal change",
            "color_evolution": "ATP gradient intensifies, nuclear concentration at periphery increases"
        },
        "methodology": "deterministic_temporal_sequence_generation",
        "llm_cost": "0 tokens"
    }

# ============================================================================
# PHASE 2.6: RHYTHMIC PRESETS + NORMALIZED PARAMETER SPACE
# ============================================================================

# 5D normalized parameter space for compositional dynamics
# Each parameter [0.0, 1.0] captures a distinct morphological axis

SLIME_MOLD_PARAMETER_NAMES = [
    "network_connectivity",   # 0.0 = isolated patches → 1.0 = dense anastomosing mesh
    "boundary_complexity",    # 0.0 = smooth circular → 1.0 = fractal dendritic edge
    "metabolic_intensity",    # 0.0 = NESS equilibrium → 1.0 = starved aggressive exploration
    "growth_dynamism",        # 0.0 = static steady-state → 1.0 = rapid protuberant expansion
    "vein_visibility"         # 0.0 = amorphous protoplasm → 1.0 = clear hierarchical network
]

# Canonical morphological states mapped to normalized coordinates
SLIME_MOLD_COORDS = {
    "acclimation_seed": {
        "network_connectivity": 0.10,
        "boundary_complexity": 0.05,
        "metabolic_intensity": 0.40,
        "growth_dynamism": 0.15,
        "vein_visibility": 0.05
    },
    "protuberant_explorer": {
        "network_connectivity": 0.35,
        "boundary_complexity": 0.85,
        "metabolic_intensity": 0.80,
        "growth_dynamism": 0.95,
        "vein_visibility": 0.30
    },
    "radial_expander": {
        "network_connectivity": 0.50,
        "boundary_complexity": 0.40,
        "metabolic_intensity": 0.55,
        "growth_dynamism": 0.65,
        "vein_visibility": 0.45
    },
    "network_architect": {
        "network_connectivity": 0.90,
        "boundary_complexity": 0.60,
        "metabolic_intensity": 0.45,
        "growth_dynamism": 0.40,
        "vein_visibility": 0.95
    },
    "starved_pathfinder": {
        "network_connectivity": 0.40,
        "boundary_complexity": 0.92,
        "metabolic_intensity": 1.00,
        "growth_dynamism": 1.00,
        "vein_visibility": 0.35
    },
    "dense_consolidated": {
        "network_connectivity": 1.00,
        "boundary_complexity": 0.15,
        "metabolic_intensity": 0.10,
        "growth_dynamism": 0.05,
        "vein_visibility": 1.00
    },
    "boundary_adapter": {
        "network_connectivity": 0.65,
        "boundary_complexity": 0.75,
        "metabolic_intensity": 0.30,
        "growth_dynamism": 0.20,
        "vein_visibility": 0.70
    },
    "fractal_peak": {
        "network_connectivity": 0.55,
        "boundary_complexity": 1.00,
        "metabolic_intensity": 0.60,
        "growth_dynamism": 0.50,
        "vein_visibility": 0.55
    }
}

# Phase 2.6 Rhythmic Presets
# Each preset oscillates between two canonical states at a characteristic period
SLIME_MOLD_RHYTHMIC_PRESETS = {
    "exploration_pulse": {
        "description": "Cyclic expansion/retraction between seed quiescence and aggressive exploration",
        "state_a": "acclimation_seed",
        "state_b": "starved_pathfinder",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 22,
        "aesthetic_character": "Breathing organism, rhythmic pseudopod advance and retreat"
    },
    "network_emergence": {
        "description": "Transition from amorphous protoplasm to hierarchical vein architecture",
        "state_a": "radial_expander",
        "state_b": "network_architect",
        "pattern": "triangular",
        "num_cycles": 2,
        "steps_per_cycle": 28,
        "aesthetic_character": "Crystallization of order from biological chaos"
    },
    "boundary_oscillation": {
        "description": "Boundary complexity cycling between smooth and fractal states",
        "state_a": "dense_consolidated",
        "state_b": "fractal_peak",
        "pattern": "sinusoidal",
        "num_cycles": 4,
        "steps_per_cycle": 18,
        "aesthetic_character": "Edge dissolving and reforming, complexity waveform"
    },
    "metabolic_tide": {
        "description": "ATP-driven metabolic wave from equilibrium to aggressive foraging",
        "state_a": "boundary_adapter",
        "state_b": "protuberant_explorer",
        "pattern": "sinusoidal",
        "num_cycles": 5,
        "steps_per_cycle": 15,
        "aesthetic_character": "Luminous metabolic pulse radiating from center to periphery"
    },
    "consolidation_cycle": {
        "description": "Alternation between explosive growth and network consolidation",
        "state_a": "protuberant_explorer",
        "state_b": "dense_consolidated",
        "pattern": "square",
        "num_cycles": 3,
        "steps_per_cycle": 20,
        "aesthetic_character": "Dramatic morphological switching between chaos and order"
    }
}


def _generate_slime_mold_oscillation(
    num_steps: int,
    num_cycles: float,
    pattern: str
) -> List[float]:
    """Generate oscillation pattern [0, 1] for Phase 2.6 presets."""
    result = []
    for i in range(num_steps):
        t = 2 * math.pi * num_cycles * i / num_steps
        if pattern == "sinusoidal":
            result.append(0.5 * (1 + math.sin(t)))
        elif pattern == "triangular":
            t_norm = (t / (2 * math.pi)) % 1.0
            result.append(2 * t_norm if t_norm < 0.5 else 2 * (1 - t_norm))
        elif pattern == "square":
            t_norm = (t / (2 * math.pi)) % 1.0
            result.append(0.0 if t_norm < 0.5 else 1.0)
        else:
            result.append(0.5)
    return result


def _generate_preset_trajectory(preset_config: dict) -> List[Dict[str, float]]:
    """Generate full trajectory for a Phase 2.6 preset as list of state dicts."""
    state_a = SLIME_MOLD_COORDS[preset_config["state_a"]]
    state_b = SLIME_MOLD_COORDS[preset_config["state_b"]]

    total_steps = preset_config["num_cycles"] * preset_config["steps_per_cycle"]
    alphas = _generate_slime_mold_oscillation(
        total_steps, preset_config["num_cycles"], preset_config["pattern"]
    )

    trajectory = []
    for alpha in alphas:
        state = {}
        for p in SLIME_MOLD_PARAMETER_NAMES:
            state[p] = round(state_a[p] * (1 - alpha) + state_b[p] * alpha, 4)
        trajectory.append(state)
    return trajectory


# ============================================================================
# PHASE 2.7: VISUAL VOCABULARY + ATTRACTOR PROMPT GENERATION
# ============================================================================

# Visual type vocabulary for nearest-neighbor matching and prompt generation
SLIME_MOLD_VISUAL_TYPES = {
    "nascent_protoplasm": {
        "coords": {
            "network_connectivity": 0.10,
            "boundary_complexity": 0.10,
            "metabolic_intensity": 0.40,
            "growth_dynamism": 0.20,
            "vein_visibility": 0.05
        },
        "keywords": [
            "translucent amoeboid mass",
            "smooth gelatinous surface",
            "faint bioluminescent glow",
            "nascent protoplasmic form",
            "minimal internal structure",
            "soft circular silhouette",
            "early-stage slime mold inoculation"
        ],
        "optical": {"finish": "subsurface_scatter", "scatter": "diffuse_glow",
                    "transparency": "translucent"},
        "color_associations": ["pale amber", "soft yellow", "translucent cream",
                               "faint bioluminescent green"]
    },
    "dendritic_explorer": {
        "coords": {
            "network_connectivity": 0.35,
            "boundary_complexity": 0.90,
            "metabolic_intensity": 0.85,
            "growth_dynamism": 0.95,
            "vein_visibility": 0.30
        },
        "keywords": [
            "aggressive dendritic pseudopod projections",
            "fractal branching boundary edge",
            "luminous ATP gradient at growth tips",
            "finger-like protrusions reaching outward",
            "irregular asymmetric sprawl",
            "high-energy exploration front",
            "Physarum pathfinding tentacles"
        ],
        "optical": {"finish": "bioluminescent_wet", "scatter": "edge_glow",
                    "transparency": "semi_translucent"},
        "color_associations": ["bright yellow", "luminous gold",
                               "electric yellow tips", "warm amber veins"]
    },
    "vein_lattice": {
        "coords": {
            "network_connectivity": 0.90,
            "boundary_complexity": 0.50,
            "metabolic_intensity": 0.40,
            "growth_dynamism": 0.35,
            "vein_visibility": 0.95
        },
        "keywords": [
            "hierarchical vein network architecture",
            "visible cytoplasmic streaming channels",
            "tree-like branching transport lattice",
            "primary and secondary vein hierarchy",
            "shuttle-streaming protoplasmic flow",
            "optimized biological transport network",
            "Physarum polycephalum vein structure"
        ],
        "optical": {"finish": "vascular_wet", "scatter": "directional_streaming",
                    "transparency": "opaque_veins"},
        "color_associations": ["deep yellow ochre", "dark amber veins",
                               "golden transport channels", "warm brown network"]
    },
    "fractal_boundary": {
        "coords": {
            "network_connectivity": 0.55,
            "boundary_complexity": 1.00,
            "metabolic_intensity": 0.60,
            "growth_dynamism": 0.50,
            "vein_visibility": 0.55
        },
        "keywords": [
            "self-similar fractal edge topology",
            "space-filling boundary complexity",
            "multi-scale dendritic detail",
            "maximum morphological irregularity",
            "fractal dimension approaching 1.92",
            "complex perimeter at peak evolution",
            "intricate biological coastline"
        ],
        "optical": {"finish": "matte_biological", "scatter": "edge_scatter",
                    "transparency": "variable_translucency"},
        "color_associations": ["variegated yellow", "fractal gold edge",
                               "mottled amber boundary", "bright dendritic tips"]
    },
    "consolidated_mesh": {
        "coords": {
            "network_connectivity": 1.00,
            "boundary_complexity": 0.15,
            "metabolic_intensity": 0.10,
            "growth_dynamism": 0.05,
            "vein_visibility": 1.00
        },
        "keywords": [
            "dense anastomosing network mesh",
            "stable consolidated vein architecture",
            "smooth outer boundary enclosing lattice",
            "redundant parallel transport paths",
            "mature NESS-state morphology",
            "equilibrium biological computation",
            "optimized slime mold steady-state"
        ],
        "optical": {"finish": "dense_matte", "scatter": "minimal_scatter",
                    "transparency": "opaque"},
        "color_associations": ["dark mature yellow", "dense ochre mesh",
                               "brownish consolidated mass", "muted amber lattice"]
    }
}


def _extract_slime_mold_visual_vocabulary(
    state: Dict[str, float],
    strength: float = 1.0
) -> Dict[str, Any]:
    """
    Map a 5D parameter state to nearest visual type and return keywords.

    Layer 2: Deterministic nearest-neighbor lookup (0 tokens).
    """
    min_dist = float('inf')
    nearest_type = None

    for type_id, type_def in SLIME_MOLD_VISUAL_TYPES.items():
        coords = type_def["coords"]
        dist_sq = sum(
            (state.get(p, 0.0) - coords.get(p, 0.0)) ** 2
            for p in SLIME_MOLD_PARAMETER_NAMES
        )
        dist = dist_sq ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest_type = type_id

    type_def = SLIME_MOLD_VISUAL_TYPES[nearest_type]
    keywords = list(type_def["keywords"])

    # Weight keywords by strength
    if strength < 1.0:
        keywords = keywords[:max(3, int(len(keywords) * strength))]

    return {
        "nearest_type": nearest_type,
        "distance": round(min_dist, 4),
        "keywords": keywords,
        "optical_properties": {
            "translucency": round(1.0 - state.get("vein_visibility", 0.5), 2),
            "bioluminescence": round(state.get("metabolic_intensity", 0.5), 2),
            "edge_detail": round(state.get("boundary_complexity", 0.5), 2),
            "internal_flow": round(
                state.get("network_connectivity", 0.5) * state.get("vein_visibility", 0.5), 2
            )
        },
        "color_associations": _get_slime_mold_colors(state)
    }


def _get_slime_mold_colors(state: Dict[str, float]) -> List[str]:
    """Derive color palette from morphological state."""
    colors = []
    metabolic = state.get("metabolic_intensity", 0.5)
    connectivity = state.get("network_connectivity", 0.5)
    complexity = state.get("boundary_complexity", 0.5)

    if metabolic > 0.7:
        colors.extend(["bright yellow-green bioluminescence", "hot white ATP glow at tips"])
    elif metabolic > 0.3:
        colors.extend(["warm amber protoplasmic glow", "golden cytoplasmic streaming"])
    else:
        colors.extend(["pale translucent yellow", "muted ochre equilibrium"])

    if connectivity > 0.7:
        colors.append("dark vein channels against bright protoplasm")
    if complexity > 0.7:
        colors.append("luminous fractal edge detail against dark background")

    return colors


def _build_slime_mold_prompt(
    state: Dict[str, float],
    style_modifier: str = ""
) -> str:
    """Build image generation prompt from parameter state."""
    vocab = _extract_slime_mold_visual_vocabulary(state, strength=1.0)
    parts = []

    if style_modifier:
        parts.append(style_modifier)

    parts.extend(vocab["keywords"])
    parts.extend(vocab["color_associations"][:2])

    # Add optical descriptors
    optical = vocab["optical_properties"]
    if optical["bioluminescence"] > 0.6:
        parts.append("intense bioluminescent illumination")
    if optical["internal_flow"] > 0.5:
        parts.append("visible internal cytoplasmic flow")
    if optical["translucency"] > 0.6:
        parts.append("semi-transparent gelatinous body")

    return ", ".join(parts)


# ============================================================================
# PHASE 2.6 TOOLS
# ============================================================================

@mcp.tool()
def get_slime_mold_parameter_space() -> Dict[str, Any]:
    """
    Get normalized 5D parameter space for compositional dynamics.

    Layer 1: Pure taxonomy lookup (0 tokens).

    Returns:
        Parameter names, canonical states, and coordinate mappings.
    """
    return {
        "domain": "slime_mold_network",
        "parameter_names": SLIME_MOLD_PARAMETER_NAMES,
        "parameter_semantics": {
            "network_connectivity": "0.0 = isolated patches → 1.0 = dense anastomosing mesh",
            "boundary_complexity": "0.0 = smooth circular → 1.0 = fractal dendritic edge",
            "metabolic_intensity": "0.0 = NESS equilibrium → 1.0 = starved aggressive exploration",
            "growth_dynamism": "0.0 = static steady-state → 1.0 = rapid protuberant expansion",
            "vein_visibility": "0.0 = amorphous protoplasm → 1.0 = clear hierarchical network"
        },
        "canonical_states": SLIME_MOLD_COORDS,
        "state_count": len(SLIME_MOLD_COORDS),
        "bounds": [0.0, 1.0],
        "methodology": "normalized_morphospace_from_empirical_taxonomy",
        "llm_cost": "0 tokens"
    }


@mcp.tool()
def list_slime_mold_rhythmic_presets() -> Dict[str, Any]:
    """
    List all Phase 2.6 rhythmic presets with periods and descriptions.

    Layer 1: Pure taxonomy lookup (0 tokens).

    Returns:
        Preset names, periods, patterns, and aesthetic descriptions.
    """
    presets_summary = {}
    for name, config in SLIME_MOLD_RHYTHMIC_PRESETS.items():
        presets_summary[name] = {
            "period": config["steps_per_cycle"],
            "total_steps": config["num_cycles"] * config["steps_per_cycle"],
            "pattern": config["pattern"],
            "states": f"{config['state_a']} ↔ {config['state_b']}",
            "description": config["description"],
            "aesthetic_character": config["aesthetic_character"]
        }

    return {
        "domain": "slime_mold_network",
        "phase": "2.6",
        "presets": presets_summary,
        "count": len(presets_summary),
        "periods": sorted(set(
            c["steps_per_cycle"] for c in SLIME_MOLD_RHYTHMIC_PRESETS.values()
        )),
        "methodology": "phase_2_6_rhythmic_preset_enumeration",
        "llm_cost": "0 tokens"
    }


@mcp.tool()
def apply_slime_mold_rhythmic_preset(preset_name: str) -> Dict[str, Any]:
    """
    Apply a Phase 2.6 rhythmic preset and return full oscillation sequence.

    Layer 2: Deterministic sequence generation (0 tokens).

    Args:
        preset_name: One of the 5 rhythmic presets
            (exploration_pulse, network_emergence, boundary_oscillation,
             metabolic_tide, consolidation_cycle)

    Returns:
        Complete oscillation sequence with parameter states at each step.
    """
    if preset_name not in SLIME_MOLD_RHYTHMIC_PRESETS:
        return {
            "error": f"Unknown preset: {preset_name}",
            "available": list(SLIME_MOLD_RHYTHMIC_PRESETS.keys())
        }

    config = SLIME_MOLD_RHYTHMIC_PRESETS[preset_name]
    trajectory = _generate_preset_trajectory(config)

    return {
        "domain": "slime_mold_network",
        "preset": preset_name,
        "description": config["description"],
        "aesthetic_character": config["aesthetic_character"],
        "pattern": config["pattern"],
        "period": config["steps_per_cycle"],
        "num_cycles": config["num_cycles"],
        "total_steps": len(trajectory),
        "states": f"{config['state_a']} ↔ {config['state_b']}",
        "sequence": trajectory,
        "parameter_names": SLIME_MOLD_PARAMETER_NAMES,
        "methodology": "phase_2_6_deterministic_oscillation",
        "llm_cost": "0 tokens"
    }


@mcp.tool()
def generate_slime_mold_rhythmic_sequence(
    state_a_id: str,
    state_b_id: str,
    oscillation_pattern: str = "sinusoidal",
    num_cycles: int = 3,
    steps_per_cycle: int = 20,
    phase_offset: float = 0.0
) -> Dict[str, Any]:
    """
    Generate custom rhythmic oscillation between two slime mold states.

    Layer 2: Temporal composition (0 tokens).

    Args:
        state_a_id: Starting canonical state
        state_b_id: Alternating canonical state
        oscillation_pattern: "sinusoidal", "triangular", or "square"
        num_cycles: Number of complete A→B→A cycles
        steps_per_cycle: Samples per cycle
        phase_offset: Starting phase (0.0 = A, 0.5 = B)

    Returns:
        Sequence with states, pattern info, and phase points.
    """
    if state_a_id not in SLIME_MOLD_COORDS:
        return {"error": f"Unknown state: {state_a_id}", "available": list(SLIME_MOLD_COORDS.keys())}
    if state_b_id not in SLIME_MOLD_COORDS:
        return {"error": f"Unknown state: {state_b_id}", "available": list(SLIME_MOLD_COORDS.keys())}
    if oscillation_pattern not in ("sinusoidal", "triangular", "square"):
        return {"error": f"Unknown pattern: {oscillation_pattern}"}

    config = {
        "state_a": state_a_id,
        "state_b": state_b_id,
        "pattern": oscillation_pattern,
        "num_cycles": num_cycles,
        "steps_per_cycle": steps_per_cycle
    }
    trajectory = _generate_preset_trajectory(config)

    # Apply phase offset by rotating trajectory
    if phase_offset > 0.0:
        offset_steps = int(phase_offset * steps_per_cycle) % len(trajectory)
        trajectory = trajectory[offset_steps:] + trajectory[:offset_steps]

    return {
        "domain": "slime_mold_network",
        "state_a": state_a_id,
        "state_b": state_b_id,
        "oscillation_pattern": oscillation_pattern,
        "period": steps_per_cycle,
        "num_cycles": num_cycles,
        "total_steps": len(trajectory),
        "phase_offset": phase_offset,
        "sequence": trajectory,
        "parameter_names": SLIME_MOLD_PARAMETER_NAMES,
        "methodology": "phase_2_6_custom_rhythmic_generation",
        "llm_cost": "0 tokens"
    }


# ============================================================================
# PHASE 2.7 TOOLS
# ============================================================================

@mcp.tool()
def list_slime_mold_visual_types() -> Dict[str, Any]:
    """
    List all visual vocabulary types for prompt generation.

    Layer 1: Pure taxonomy lookup (0 tokens).

    Returns:
        Visual types with coordinates, keywords, and aesthetic descriptions.
    """
    summary = {}
    for type_id, type_def in SLIME_MOLD_VISUAL_TYPES.items():
        summary[type_id] = {
            "coords": type_def["coords"],
            "keyword_count": len(type_def["keywords"]),
            "keywords_preview": type_def["keywords"][:3]
        }

    return {
        "domain": "slime_mold_network",
        "phase": "2.7",
        "visual_types": summary,
        "count": len(summary),
        "parameter_names": SLIME_MOLD_PARAMETER_NAMES,
        "methodology": "visual_vocabulary_taxonomy",
        "llm_cost": "0 tokens"
    }


@mcp.tool()
def extract_slime_mold_visual_vocabulary(
    state: Dict[str, float],
    strength: float = 1.0
) -> Dict[str, Any]:
    """
    Extract visual vocabulary from slime mold parameter coordinates.

    Layer 2: Deterministic vocabulary mapping (0 tokens).

    Maps a 5D parameter state to the nearest visual type and returns
    image-generation-ready keywords.

    Args:
        state: Parameter coordinates dict with keys:
            network_connectivity, boundary_complexity, metabolic_intensity,
            growth_dynamism, vein_visibility
        strength: Keyword weight multiplier [0.0, 1.0] (default: 1.0)

    Returns:
        Nearest visual type, keywords, optical properties, and color palette.
    """
    missing = [p for p in SLIME_MOLD_PARAMETER_NAMES if p not in state]
    if missing:
        return {"error": f"Missing parameters: {missing}", "required": SLIME_MOLD_PARAMETER_NAMES}

    vocab = _extract_slime_mold_visual_vocabulary(state, strength)

    return {
        "domain": "slime_mold_network",
        "input_state": state,
        "nearest_type": vocab["nearest_type"],
        "distance": vocab["distance"],
        "keywords": vocab["keywords"],
        "optical_properties": vocab["optical_properties"],
        "color_associations": vocab["color_associations"],
        "methodology": "nearest_neighbor_vocabulary_extraction",
        "llm_cost": "0 tokens"
    }


@mcp.tool()
def generate_slime_mold_prompt(
    state_id: str = "",
    custom_state: Optional[Dict[str, float]] = None,
    mode: str = "composite",
    style_modifier: str = ""
) -> Dict[str, Any]:
    """
    Generate image generation prompt from slime mold state or coordinates.

    Layer 2: Deterministic prompt synthesis (0 tokens).

    Translates slime mold morphological coordinates into visual prompts
    suitable for ComfyUI, Stable Diffusion, DALL-E, etc.

    Args:
        state_id: Canonical state name (or "" with custom_state)
        custom_state: Optional custom 5D coordinates
        mode: "composite" (single blended prompt) or "split_view" (per-category)
        style_modifier: Optional prefix ("photorealistic", "fluorescence microscopy", etc.)

    Returns:
        Prompt string(s) with vocabulary details and state metadata.
    """
    # Resolve state
    if custom_state:
        state = custom_state
        source = "custom_state"
    elif state_id and state_id in SLIME_MOLD_COORDS:
        state = SLIME_MOLD_COORDS[state_id]
        source = state_id
    else:
        return {
            "error": "Provide state_id or custom_state",
            "available_states": list(SLIME_MOLD_COORDS.keys())
        }

    vocab = _extract_slime_mold_visual_vocabulary(state, strength=1.0)

    if mode == "composite":
        prompt = _build_slime_mold_prompt(state, style_modifier)
        return {
            "domain": "slime_mold_network",
            "source": source,
            "mode": "composite",
            "prompt": prompt,
            "vocabulary": vocab,
            "methodology": "phase_2_7_composite_prompt_synthesis",
            "llm_cost": "0 tokens"
        }

    elif mode == "split_view":
        prompts = {}

        # Network structure category
        if state.get("vein_visibility", 0) > 0.6:
            net_kw = [
                "hierarchical vein network", "branching transport channels",
                "shuttle-streaming cytoplasm", "optimized biological lattice"
            ]
        else:
            net_kw = [
                "amorphous protoplasmic mass", "nascent flow channels",
                "pre-network biological medium", "diffuse cytoplasmic body"
            ]

        # Boundary form category
        if state.get("boundary_complexity", 0) > 0.6:
            bound_kw = [
                "fractal dendritic boundary", "self-similar edge detail",
                "pseudopod projections", "irregular branching perimeter"
            ]
        else:
            bound_kw = [
                "smooth circular boundary", "clean geometric silhouette",
                "minimal surface irregularity", "contained spheroid form"
            ]

        # Metabolic energy category
        if state.get("metabolic_intensity", 0) > 0.6:
            meta_kw = [
                "bright bioluminescent ATP gradient", "intense peripheral glow",
                "high-energy metabolic activity", "luminous growth-front radiance"
            ]
        else:
            meta_kw = [
                "soft ambient bioluminescence", "steady-state metabolic glow",
                "equilibrium energy distribution", "muted protoplasmic warmth"
            ]

        for cat_name, cat_kw in [
            ("network_structure", net_kw),
            ("boundary_form", bound_kw),
            ("metabolic_energy", meta_kw)
        ]:
            if style_modifier:
                cat_kw = [style_modifier] + cat_kw
            prompts[cat_name] = ", ".join(cat_kw)

        return {
            "domain": "slime_mold_network",
            "source": source,
            "mode": "split_view",
            "prompts": prompts,
            "vocabulary": vocab,
            "methodology": "phase_2_7_split_view_prompt_synthesis",
            "llm_cost": "0 tokens"
        }

    else:
        return {"error": f"Unknown mode: {mode}", "available": ["composite", "split_view"]}


@mcp.tool()
def generate_slime_mold_sequence_prompts(
    preset_name: str,
    keyframe_count: int = 4,
    style_modifier: str = ""
) -> Dict[str, Any]:
    """
    Generate keyframe prompts from a Phase 2.6 rhythmic preset.

    Layer 2: Deterministic keyframe extraction (0 tokens).

    Extracts evenly-spaced keyframes from a rhythmic oscillation
    and generates an image prompt for each. Useful for storyboards,
    animation keyframes, and multi-panel visualizations.

    Args:
        preset_name: One of the 5 rhythmic presets
        keyframe_count: Number of keyframes to extract (default: 4)
        style_modifier: Optional style prefix for all prompts

    Returns:
        Keyframes with step index, state, prompt, and vocabulary.
    """
    if preset_name not in SLIME_MOLD_RHYTHMIC_PRESETS:
        return {
            "error": f"Unknown preset: {preset_name}",
            "available": list(SLIME_MOLD_RHYTHMIC_PRESETS.keys())
        }

    config = SLIME_MOLD_RHYTHMIC_PRESETS[preset_name]
    trajectory = _generate_preset_trajectory(config)
    total_steps = len(trajectory)

    keyframes = []
    for k in range(keyframe_count):
        step_idx = int(k * total_steps / keyframe_count) % total_steps
        state = trajectory[step_idx]
        prompt = _build_slime_mold_prompt(state, style_modifier)
        vocab = _extract_slime_mold_visual_vocabulary(state)

        keyframes.append({
            "keyframe": k,
            "step": step_idx,
            "state": state,
            "prompt": prompt,
            "nearest_visual_type": vocab["nearest_type"],
            "distance_to_type": vocab["distance"]
        })

    return {
        "domain": "slime_mold_network",
        "preset": preset_name,
        "description": config["description"],
        "aesthetic_character": config["aesthetic_character"],
        "period": config["steps_per_cycle"],
        "total_steps": total_steps,
        "keyframe_count": keyframe_count,
        "keyframes": keyframes,
        "methodology": "phase_2_7_sequence_prompt_extraction",
        "llm_cost": "0 tokens"
    }


# ============================================================================
# PHASE 2.8: AESTHETIC DECOMPOSITION — description → 5D coordinates
# ============================================================================
# Inverse of the generative pipeline: text description → domain coordinates.
# Completes the round-trip: coordinates → prompt → image → description → coords.
# Layer 2: deterministic, 0 LLM tokens.

import re as _re
import math as _math

_DECOMPOSE_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'in', 'on', 'at', 'to', 'of', 'for', 'with',
    'by', 'from', 'and', 'or', 'but', 'as', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'has', 'have', 'had', 'do', 'does', 'did',
    'no', 'not', 'all', 'its', 'this', 'that', 'into', 'over',
})


def _decompose_tokenize(text: str):
    """Tokenize input text into word set + lowercased full text."""
    lower = text.lower()
    words = set(_re.findall(r'[a-z]+(?:-[a-z]+)*', lower))
    return words, lower


def _decompose_extract_fragments(keyword: str) -> List[str]:
    """Extract matchable sub-phrases from a keyword string.

    Sliding window of 2-4 words + full keyword. Skips stop-word-only frags.
    """
    words = keyword.lower().split()
    fragments = []
    if len(words) >= 3:
        fragments.append(keyword.lower())
    for window_size in [4, 3, 2]:
        for i in range(len(words) - window_size + 1):
            frag = ' '.join(words[i:i + window_size])
            content_words = [w for w in words[i:i + window_size]
                             if len(w) > 3 and w not in _DECOMPOSE_STOP_WORDS]
            if content_words:
                fragments.append(frag)
    return fragments


def _decompose_score_visual_type(
    type_def: dict,
    words: set,
    full_text: str,
    sub_w: float = 1.0,
    word_w: float = 0.3,
    opt_w: float = 0.5,
    color_w: float = 0.4,
) -> tuple:
    """Score a single visual type against input text. Returns (score, matches)."""
    score = 0.0
    matched = []

    # Keyword fragment matching
    for keyword in type_def.get("keywords", []):
        fragments = _decompose_extract_fragments(keyword)
        best_s, best_f = 0.0, None
        for frag in fragments:
            if frag in full_text:
                if sub_w > best_s:
                    best_s, best_f = sub_w, frag
            else:
                frag_words = set(frag.split()) - _DECOMPOSE_STOP_WORDS
                if frag_words:
                    overlap = len(frag_words & words) / len(frag_words)
                    ws = overlap * word_w
                    if ws > best_s:
                        best_s, best_f = ws, frag
        if best_f and best_s > 0:
            score += best_s
            matched.append(best_f)

    # Optical property matching
    for prop_name, prop_value in type_def.get("optical", {}).items():
        prop_words = set(prop_value.lower().replace('_', ' ').split())
        prop_overlap = len(prop_words & words)
        if prop_overlap > 0:
            score += opt_w * (prop_overlap / len(prop_words))
            matched.append(f"optical:{prop_value}")

    # Color association matching
    for color in type_def.get("color_associations", []):
        color_lower = color.lower()
        if color_lower in full_text:
            score += color_w
            matched.append(f"color:{color}")
        else:
            color_words = set(color_lower.split()) - _DECOMPOSE_STOP_WORDS
            if color_words:
                overlap = len(color_words & words)
                if overlap > 0:
                    score += color_w * 0.5 * (overlap / len(color_words))

    return score, matched


def _decompose_slime_mold(description: str) -> Dict[str, Any]:
    """Core decomposition: text description → 5D slime mold coordinates.

    Layer 2: deterministic, 0 LLM tokens.

    Algorithm:
      1. Tokenize description
      2. Score each visual type by keyword/optical/color matching
      3. Softmax-blend type centers → recovered coordinates
      4. Compute confidence from max score vs max possible
      5. Return coordinates + metadata
    """
    words, full_text = _decompose_tokenize(description)

    # Score each visual type
    type_scores = {}
    all_matched = []
    for type_id, type_def in SLIME_MOLD_VISUAL_TYPES.items():
        score, matched = _decompose_score_visual_type(type_def, words, full_text)
        type_scores[type_id] = score
        all_matched.extend(matched)

    max_score = max(type_scores.values()) if type_scores else 0
    # Max possible: ~7 keywords × 1.0 + 3 optical × 0.5 + 4 colors × 0.4 = 10.1
    max_possible = 10.1
    confidence = min(1.0, max_score / max_possible) if max_possible > 0 else 0.0

    # Below threshold → domain not detected
    if confidence < 0.05:
        return {
            "detected": False,
            "coordinates": {p: 0.5 for p in SLIME_MOLD_PARAMETER_NAMES},
            "confidence": 0.0,
            "nearest_type": "",
            "type_scores": type_scores,
        }

    # Softmax blend coordinates (temperature=1.5)
    temp = 1.5
    max_s = max(type_scores.values())
    exps = {k: _math.exp((v - max_s) / temp) for k, v in type_scores.items()}
    total_exp = sum(exps.values())
    weights = {k: v / total_exp for k, v in exps.items()}

    coords = {p: 0.0 for p in SLIME_MOLD_PARAMETER_NAMES}
    for type_id, w in weights.items():
        if w > 0:
            center = SLIME_MOLD_VISUAL_TYPES[type_id]["coords"]
            for p in SLIME_MOLD_PARAMETER_NAMES:
                coords[p] += w * center.get(p, 0)

    # Round coordinates
    coords = {p: round(v, 4) for p, v in coords.items()}

    # Nearest type
    nearest_type = max(type_scores, key=type_scores.get)
    nearest_center = SLIME_MOLD_VISUAL_TYPES[nearest_type]["coords"]
    nearest_dist = _math.sqrt(sum(
        (coords.get(p, 0) - nearest_center.get(p, 0)) ** 2
        for p in SLIME_MOLD_PARAMETER_NAMES
    ))

    # Separate matched fragments from optical/color
    optical_matches = {}
    color_matches = []
    keyword_matches = []
    for m in all_matched:
        if m.startswith("optical:"):
            val = m.split(":", 1)[1]
            optical_matches[val] = True
        elif m.startswith("color:"):
            color_matches.append(m.split(":", 1)[1])
        else:
            if m not in keyword_matches:
                keyword_matches.append(m)

    return {
        "detected": True,
        "coordinates": coords,
        "confidence": round(confidence, 4),
        "nearest_type": nearest_type,
        "nearest_type_distance": round(nearest_dist, 4),
        "type_scores": {k: round(v, 3) for k, v in type_scores.items()},
        "type_weights": {k: round(v, 4) for k, v in weights.items()},
        "matched_fragments": keyword_matches[:10],
        "optical_matches": list(optical_matches.keys()),
        "color_matches": color_matches,
    }


@mcp.tool()
def decompose_slime_mold_from_description(description: str) -> Dict[str, Any]:
    """
    Decompose a text description into slime mold 5D coordinates.

    Layer 2: Deterministic keyword matching (0 LLM tokens).

    Inverse of the generative pipeline: takes an image description
    (from Claude vision, user text, or any aesthetic description)
    and recovers the slime mold morphological coordinates that would
    produce similar visual vocabulary.

    Algorithm:
      1. Tokenize description into words and bigrams
      2. Score each visual type by keyword fragment matching,
         optical property overlap, and color association hits
      3. Softmax-blend type center coordinates (temperature=1.5)
      4. Return recovered 5D coordinates with confidence

    Confidence is NOT "how good the image is" — it's "how much
    slime mold visual vocabulary is present in the description."

    Args:
        description: Image description text or aesthetic description.
            Examples:
            - "translucent amoeboid mass with faint bioluminescent glow"
            - "hierarchical vein network with branching transport channels"
            - "fractal dendritic boundary with self-similar edge detail"
            - "dense anastomosing network mesh at steady state"

    Returns:
        Recovered 5D coordinates, confidence, nearest visual type,
        matched fragments, and scoring metadata.

    Cost: 0 tokens (pure Layer 2 deterministic computation)
    """
    if not description or not description.strip():
        return {"error": "Empty description", "usage": "Provide text describing a slime mold aesthetic"}

    result = _decompose_slime_mold(description)

    return {
        "domain": "slime_mold_network",
        "phase": "2.8",
        "description_length": len(description),
        "detected": result["detected"],
        "coordinates": result["coordinates"],
        "confidence": result.get("confidence", 0.0),
        "nearest_type": result.get("nearest_type", ""),
        "nearest_type_distance": result.get("nearest_type_distance", 0.0),
        "type_scores": result.get("type_scores", {}),
        "type_weights": result.get("type_weights", {}),
        "matched_fragments": result.get("matched_fragments", []),
        "optical_matches": result.get("optical_matches", []),
        "color_matches": result.get("color_matches", []),
        "parameter_names": SLIME_MOLD_PARAMETER_NAMES,
        "methodology": "phase_2_8_aesthetic_decomposition",
        "algorithm": "keyword_fragment_matching_softmax_blend",
        "llm_cost": "0 tokens",
    }


@mcp.tool()
def get_slime_mold_server_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the Slime Mold Network MCP server.

    Returns server metadata, capabilities, and phase status.
    """
    return {
        "server": "slime-mold-network",
        "version": "2.8.0",
        "description": "Morphological computation aesthetics based on Physarum polycephalum",
        "research_basis": "Bajpai et al. (2025)",
        "layer_architecture": {
            "layer_1": "Pure taxonomy lookup (growth phases, topologies, strains, boundaries, scales)",
            "layer_2": "Deterministic mappings (morphology, ATP, bounds, oscillations, vocabulary)",
            "layer_3": "Synthesis preparation (visualization, temporal sequences, prompts)"
        },
        "phase_2_6_enhancements": {
            "rhythmic_presets": True,
            "preset_count": len(SLIME_MOLD_RHYTHMIC_PRESETS),
            "presets": {
                name: {
                    "period": c["steps_per_cycle"],
                    "pattern": c["pattern"],
                    "states": f"{c['state_a']} ↔ {c['state_b']}"
                }
                for name, c in SLIME_MOLD_RHYTHMIC_PRESETS.items()
            },
            "periods": sorted(set(
                c["steps_per_cycle"] for c in SLIME_MOLD_RHYTHMIC_PRESETS.values()
            )),
            "parameter_space": {
                "dimensions": len(SLIME_MOLD_PARAMETER_NAMES),
                "parameters": SLIME_MOLD_PARAMETER_NAMES,
                "canonical_states": len(SLIME_MOLD_COORDS)
            }
        },
        "phase_2_7_enhancements": {
            "attractor_visualization": True,
            "visual_type_count": len(SLIME_MOLD_VISUAL_TYPES),
            "visual_types": list(SLIME_MOLD_VISUAL_TYPES.keys()),
            "prompt_modes": ["composite", "split_view"],
            "sequence_prompts": True
        },
        "phase_2_8_enhancements": {
            "aesthetic_decomposition": True,
            "description": "Text description → 5D coordinate recovery",
            "algorithm": "keyword_fragment_matching_softmax_blend",
            "visual_types_enriched": True,
            "optical_properties_added": True,
            "color_associations_added": True,
            "tool": "decompose_slime_mold_from_description",
            "cost": "0 tokens (Layer 2 deterministic)"
        },
        "compositional_integration": {
            "ready_for_tier_4d": True,
            "forced_orbit_compatible": True,
            "domain_registry_format": {
                "domain_id": "slime_mold",
                "parameter_names": SLIME_MOLD_PARAMETER_NAMES,
                "preset_periods": sorted(set(
                    c["steps_per_cycle"] for c in SLIME_MOLD_RHYTHMIC_PRESETS.values()
                ))
            }
        },
        "llm_cost": "0 tokens (all Layer 1-2 operations)"
    }


# ============================================================================
# EXISTING TOOLS (preserved)
# ============================================================================

@mcp.tool()
def suggest_composition_domains() -> Dict[str, Any]:
    """
    Suggest other domains that compose well with slime mold networks.
    
    Returns recommendations for categorical composition.
    """
    return {
        "domain": "slime_mold_network",
        "composes_well_with": {
            "fractal_morph": {
                "rationale": "Slime mold boundary has fractal dimension 1.7-1.92; fractal-morph can enhance",
                "functor": "growth_phase → fractal base type",
                "example": "network_formation → dragon_curve (self-contacting paths)",
                "preserves": "Self-similar structure, space-filling tendency"
            },
            "diatom_morphology": {
                "rationale": "Both microscopic organisms with distinct structural principles",
                "functor": "network_topology → diatom frustule symmetry",
                "example": "connected_lattice → radial_array (hierarchical branching)",
                "preserves": "Biological optimization, transport network structure"
            },
            "catastrophe_morph": {
                "rationale": "Growth phase transitions = catastrophe morphologies",
                "functor": "Circularity decay → fold catastrophe, NESS → cusp catastrophe",
                "example": "boundary_adaptation → butterfly (4 control parameters)",
                "preserves": "Sudden morphological transitions, critical points"
            },
            "nuclear_aesthetic": {
                "rationale": "ATP gradient ↔ thermal energy gradient, both center-to-periphery flows",
                "functor": "metabolic_state → nuclear phase",
                "example": "starved_aggressive → expansion_front (rapid growth seeking energy)",
                "preserves": "Energy distribution patterns, boundary propagation"
            },
            "origami_aesthetics": {
                "rationale": "Vein network = crease pattern, complexity parameter analogous",
                "functor": "network_topology → crease pattern complexity",
                "example": "dense_mesh → tessellation (repeated modular structure)",
                "preserves": "Geometric constraints, structural optimization"
            }
        },
        "composition_principles": {
            "categorical_compatibility": "All suggested domains have independent dimensional structure",
            "morphism_preservation": "growth_phase and network_topology are morphisms (arrows)",
            "object_stability": "strain_profile is object (node) - stable under composition",
            "constraint_interaction": "boundary_character (constraint) composes with fractal recursion depth"
        }
    }

@mcp.tool()
def get_research_attribution() -> Dict[str, Any]:
    """
    Get complete research citations and educational resources.
    
    Returns attribution to Levin, Kurian, Nakagaki and related research.
    """
    return {
        "domain": "slime_mold_network",
        "primary_paper": {
            "title": "Morphological computational capacity of Physarum polycephalum",
            "authors": "S. Bajpai, A. Lucas-DeMott, N. J. Murugan, M. Levin, P. Kurian",
            "year": 2025,
            "arxiv": "2510.19976v1 [quant-ph]",
            "institutions": [
                "Allen Discovery Center, Tufts University",
                "Quantum Biology Laboratory, Howard University",
                "Wilfrid Laurier University"
            ],
            "key_contribution": "First framework for bounding computational capacities of aneural organism based on morphology"
        },
        "foundational_work": {
            "problem_solving": {
                "author": "T. Nakagaki et al.",
                "papers": [
                    "Maze-solving by an amoeboid organism (Nature 2000)",
                    "Minimum-risk path finding by an adaptive amoebal network (PRL 2007)"
                ],
                "contribution": "Demonstrated Physarum solves TSP, maze problems, optimizes networks"
            },
            "quantum_biology": {
                "author": "P. Kurian et al.",
                "papers": [
                    "Ultraviolet superradiance from mega-networks of tryptophan (JPC B 2024)",
                    "Computational capacity of life in relation to the universe (Science Advances 2025)"
                ],
                "contribution": "Superradiant states in protein architectures, quantum optical bounds"
            },
            "morphological_computation": {
                "author": "M. Levin et al.",
                "papers": [
                    "Mechanosensation mediates long-range spatial decision-making (Adv Materials 2021)"
                ],
                "contribution": "Aneural information processing, distributed cognition"
            }
        },
        "theoretical_foundations": {
            "margolus_levitin_theorem": "N. Margolus, L.B. Levitin - Maximum speed of dynamical evolution (Physica D 1998)",
            "allometric_scaling": "G.B. West et al. - Allometric scaling of metabolic rate (PNAS 2002)",
            "fractal_biology": "G.B. West, J.H. Brown - Fourth dimension of life: fractal geometry (Science 1999)"
        },
        "experimental_methods": {
            "strains": {
                "japanese_sonobe": "Provided by Dr. Yukinori Nishigami and Prof. Toshiyuki Nakagaki",
                "carolina": "Carolina Biologicals"
            },
            "imaging": "High-throughput flatbed scanning platform (1600 dpi, 30-min intervals, up to 72h)",
            "analysis": "PyPETANA 2.0 - Custom morphological analysis pipeline"
        },
        "educational_resources": {
            "slime_mold_basics": "https://en.wikipedia.org/wiki/Physarum_polycephalum",
            "morphological_computation": "Müller & Hoffmann - What is morphological computation? (Artificial Life 2017)",
            "quantum_biology": "Kurian Lab - https://www.philkurian.com/",
            "levin_lab": "https://ase.tufts.edu/biology/labs/levin/"
        },
        "license": "Research attribution required for commercial use of this taxonomy"
    }

if __name__ == "__main__":
    mcp.run()
