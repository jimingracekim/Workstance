"""
Heavy-Lift Multirotor Feasibility Model
Workstance first-responder heavy-lift study.

"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

# 1. Physical constants

G = 9.80665          # gravitational acceleration [m/s^2]
RHO_SL = 1.225       # ISA sea-level density [kg/m^3]
INCH = 0.0254        # metres per inch
LB_TO_KG = 0.45359237


def air_density(altitude_m: float) -> float:
    if altitude_m < 0:
        raise ValueError("Altitude must be non-negative.")
    if altitude_m > 11000:
        raise ValueError("This simple ISA function is limited to 0-11 km.")
    return RHO_SL * (1 - 2.25577e-5 * altitude_m) ** 4.2559


# 2. Rotor physics


def effective_disk_area(propeller_count: int,
                        propeller_diameter_m: float,
                        coaxial: bool) -> float:
    if propeller_count <= 0 or propeller_diameter_m <= 0:
        raise ValueError("Propeller count and diameter must be positive.")
    if coaxial and propeller_count % 2:
        raise ValueError("Coaxial layouts require an even propeller count.")

    disk_count = propeller_count / 2 if coaxial else propeller_count
    return disk_count * math.pi * (propeller_diameter_m / 2) ** 2


def ideal_hover_power_w(thrust_n: float,
                        disk_area_m2: float,
                        rho: float) -> float:
    if thrust_n <= 0 or disk_area_m2 <= 0 or rho <= 0:
        raise ValueError("Thrust, disk area, and density must be positive.")
    return thrust_n ** 1.5 / math.sqrt(2 * rho * disk_area_m2)

# 3. Reference aircraft and calibration points


@dataclass(frozen=True)
class ReferenceAircraft:

    name: str = "DJI FlyCart 30"

    mtow_kg: float = 95.0
    dry_mass_kg: float = 42.5
    mass_with_batteries_kg: float = 65.0
    payload_kg: float = 30.0

    propeller_count: int = 8
    coaxial: bool = True
    propeller_diameter_in: float = 54.0

    battery_count: int = 2
    battery_energy_wh_each: float = 1984.4

    # Calibration point 1: maximum-weight hover
    hover_minutes_at_mtow: float = 18.0

    # Calibration point 2: maximum-weight range at quoted constant speed
    max_range_km_at_mtow: float = 16.0
    range_test_speed_mps: float = 15.0

    # HELD OUT from fitting and used as an independent cross-check
    empty_hover_minutes: float = 29.0
    empty_max_range_km: float = 28.0

    @property
    def battery_mass_kg(self) -> float:
        return self.mass_with_batteries_kg - self.dry_mass_kg

    @property
    def total_battery_energy_wh(self) -> float:
        return self.battery_count * self.battery_energy_wh_each

    @property
    def battery_specific_energy_wh_kg(self) -> float:
        return self.total_battery_energy_wh / self.battery_mass_kg

    @property
    def propeller_diameter_m(self) -> float:
        return self.propeller_diameter_in * INCH


# 4. Design assumptions and mission definitions


@dataclass(frozen=True)
class DesignAssumptions:
    # Propulsion mass model
    propulsion_power_density_kw_kg: float = 3.0
    motor_power_margin: float = 1.50
    maneuver_thrust_margin: float = 1.30
    require_omi: bool = True

    # Battery
    battery_specific_energy_wh_kg: float | None = None
    usable_battery_fraction: float = 0.90
    reserve_fraction: float = 0.20

    # Mass model
    fixed_system_mass_kg: float = 5.0
    structure_mass_exponent: float = 1.10

    # Geometry
    rotor_diameter_scaling_exponent: float = 1 / 3
    max_vehicle_span_m: float = 4.0
    coaxial_span_factor: float = 2.20
    independent_span_factor: float = 3.00

    # Mission
    cruise_speed_mps: float = 15.0
    scene_hover_minutes: float = 2.0
    climb_height_m: float = 200.0
    climb_rate_mps: float = 3.0
    climb_incremental_efficiency: float = 0.70
    site_altitude_m: float = 0.0


@dataclass(frozen=True)
class MissionProfile:
    name: str
    outbound_payload_kg: float
    inbound_payload_kg: float
    scene_hover_payload_kg: float
    radius_km: float
    human_scale_payload_case: bool = False

    @property
    def max_payload_kg(self) -> float:
        return max(self.outbound_payload_kg,
                   self.inbound_payload_kg,
                   self.scene_hover_payload_kg)


@dataclass(frozen=True)
class Calibration:
    hover_correction_factor: float
    cruise_power_factor: float
    structure_mass_ref_kg: float
    battery_specific_energy_wh_kg: float
    reference_disk_area_m2: float
    reference_propeller_diameter_m: float
    reference_hover_power_kw: float
    reference_mtow_kg: float
    reference_propulsion_mass_kg: float


# 5. Calibration and independent cross-check

def calibrate(ref: ReferenceAircraft,
              a: DesignAssumptions,
              verbose: bool = True) -> Calibration:

    rho = air_density(0.0)
    area = effective_disk_area(
        ref.propeller_count, ref.propeller_diameter_m, ref.coaxial
    )
    thrust_n = ref.mtow_kg * G
    ideal_hover_kw = ideal_hover_power_w(thrust_n, area, rho) / 1000

    hover_electrical_kw = (
        ref.total_battery_energy_wh / 1000
    ) / (ref.hover_minutes_at_mtow / 60)
    hover_correction = hover_electrical_kw / ideal_hover_kw

    range_time_h = (
        ref.max_range_km_at_mtow * 1000
        / ref.range_test_speed_mps
        / 3600
    )
    range_average_kw = (ref.total_battery_energy_wh / 1000) / range_time_h
    cruise_factor = range_average_kw / hover_electrical_kw

    # Propulsion sizing at the reference point.
    maneuver_power_ratio = a.maneuver_thrust_margin ** 1.5
    omi_power_ratio = (
        (ref.propeller_count / (ref.propeller_count - 1)) ** 1.5
        if a.require_omi else 1.0
    )
    required_motor_power_ratio = max(maneuver_power_ratio, omi_power_ratio)
    design_power_ratio = max(a.motor_power_margin, required_motor_power_ratio)

    propulsion_mass_kg = (
        hover_electrical_kw * design_power_ratio
        / a.propulsion_power_density_kw_kg
    )

    structure_mass_ref_kg = (
        ref.mtow_kg
        - ref.payload_kg
        - ref.battery_mass_kg
        - propulsion_mass_kg
        - a.fixed_system_mass_kg
    )
    if structure_mass_ref_kg <= 0:
        raise ValueError(
            "Reference dry-mass residual is non-positive. Check propulsion "
            "power density, margins, or fixed-system mass."
        )

    specific_energy = (
        a.battery_specific_energy_wh_kg
        if a.battery_specific_energy_wh_kg is not None
        else ref.battery_specific_energy_wh_kg
    )

    if verbose:
        print("=" * 78)
        print(f"CALIBRATION -- {ref.name}")
        print("=" * 78)
        print(f"Reference disk area               : {area:8.3f} m^2")
        print(f"Reference disk loading            : {thrust_n / area:8.1f} N/m^2")
        print(f"Ideal induced hover power         : {ideal_hover_kw:8.2f} kW")
        print(f"Empirical hover electrical power  : {hover_electrical_kw:8.2f} kW")
        print(f"Hover correction factor           : {hover_correction:8.3f}")
        print(f"Calibrated cruise / hover factor  : {cruise_factor:8.3f}")
        print(f"Battery specific energy           : {specific_energy:8.1f} Wh/kg")
        print(f"Reference propulsion mass model   : {propulsion_mass_kg:8.2f} kg")
        print(f"Reference structural residual     : {structure_mass_ref_kg:8.2f} kg")
        print()

    return Calibration(
        hover_correction_factor=hover_correction,
        cruise_power_factor=cruise_factor,
        structure_mass_ref_kg=structure_mass_ref_kg,
        battery_specific_energy_wh_kg=specific_energy,
        reference_disk_area_m2=area,
        reference_propeller_diameter_m=ref.propeller_diameter_m,
        reference_hover_power_kw=hover_electrical_kw,
        reference_mtow_kg=ref.mtow_kg,
        reference_propulsion_mass_kg=propulsion_mass_kg,
    )


def cross_validate_reference(ref: ReferenceAircraft,
                             cal: Calibration) -> dict:

    rho = air_density(0.0)
    mass_kg = ref.mass_with_batteries_kg
    thrust_n = mass_kg * G
    hover_power_kw = (
        cal.hover_correction_factor
        * ideal_hover_power_w(thrust_n, cal.reference_disk_area_m2, rho)
        / 1000
    )

    predicted_hover_minutes = (
        ref.total_battery_energy_wh / 1000 / hover_power_kw * 60
    )
    hover_error_pct = 100 * (
        predicted_hover_minutes - ref.empty_hover_minutes
    ) / ref.empty_hover_minutes

    cruise_power_kw = cal.cruise_power_factor * hover_power_kw
    predicted_range_time_h = (ref.total_battery_energy_wh / 1000) / cruise_power_kw
    predicted_range_km = predicted_range_time_h * ref.range_test_speed_mps * 3.6
    range_error_pct = 100 * (
        predicted_range_km - ref.empty_max_range_km
    ) / ref.empty_max_range_km

    print("=" * 78)
    print("INDEPENDENT CROSS-CHECKS -- HELD-OUT FLYCART 30 DATA")
    print("=" * 78)
    print(f"Published empty hover endurance : {ref.empty_hover_minutes:7.2f} min")
    print(f"Predicted empty hover endurance : {predicted_hover_minutes:7.2f} min")
    print(f"Hover prediction error          : {hover_error_pct:+7.2f} %")
    print()
    print(f"Published empty max range       : {ref.empty_max_range_km:7.2f} km")
    print(f"Predicted empty max range       : {predicted_range_km:7.2f} km")
    print(f"Range prediction error          : {range_error_pct:+7.2f} %")
    print()

    return {
        "hover_published_minutes": ref.empty_hover_minutes,
        "hover_predicted_minutes": predicted_hover_minutes,
        "hover_error_pct": hover_error_pct,
        "range_published_km": ref.empty_max_range_km,
        "range_predicted_km": predicted_range_km,
        "range_error_pct": range_error_pct,
    }


# 6. Geometry and mass models


def vehicle_geometry(max_gross_mass_kg: float,
                     cal: Calibration,
                     a: DesignAssumptions,
                     propeller_count: int,
                     coaxial: bool) -> dict:


    if max_gross_mass_kg <= 0:
        raise ValueError("Mass must be positive.")

    scaled_diameter = cal.reference_propeller_diameter_m * (
        max_gross_mass_kg / cal.reference_mtow_kg
    ) ** a.rotor_diameter_scaling_exponent

    span_factor = a.coaxial_span_factor if coaxial else a.independent_span_factor
    max_diameter_from_span = a.max_vehicle_span_m / span_factor
    diameter_m = min(scaled_diameter, max_diameter_from_span)

    area_m2 = effective_disk_area(propeller_count, diameter_m, coaxial)
    span_m = diameter_m * span_factor

    return {
        "diameter_m": diameter_m,
        "unconstrained_diameter_m": scaled_diameter,
        "area_m2": area_m2,
        "span_m": span_m,
        "span_limited": scaled_diameter > max_diameter_from_span,
    }


def structure_mass_kg(max_gross_mass_kg: float,
                      cal: Calibration,
                      a: DesignAssumptions) -> float:
    return cal.structure_mass_ref_kg * (
        max_gross_mass_kg / cal.reference_mtow_kg
    ) ** a.structure_mass_exponent


def electrical_hover_power_kw(mass_kg: float,
                              disk_area_m2: float,
                              rho: float,
                              cal: Calibration) -> float:
    return (
        cal.hover_correction_factor
        * ideal_hover_power_w(mass_kg * G, disk_area_m2, rho)
        / 1000
    )


# 7. Mission energy model

def cruise_energy_kwh(mass_kg: float,
                      distance_km: float,
                      disk_area_m2: float,
                      rho: float,
                      cal: Calibration,
                      a: DesignAssumptions) -> tuple[float, float]:
    hover_kw = electrical_hover_power_kw(mass_kg, disk_area_m2, rho, cal)
    cruise_kw = cal.cruise_power_factor * hover_kw
    time_h = distance_km * 1000 / a.cruise_speed_mps / 3600
    return cruise_kw * time_h, cruise_kw


def climb_energy_kwh(mass_kg: float,
                     disk_area_m2: float,
                     rho: float,
                     cal: Calibration,
                     a: DesignAssumptions) -> float:
    if a.climb_height_m <= 0:
        return 0.0
    if a.climb_rate_mps <= 0 or a.climb_incremental_efficiency <= 0:
        raise ValueError("Climb rate and climb efficiency must be positive.")

    hover_kw = electrical_hover_power_kw(mass_kg, disk_area_m2, rho, cal)
    climb_time_s = a.climb_height_m / a.climb_rate_mps
    potential_kw = (
        mass_kg * G * a.climb_rate_mps
        / a.climb_incremental_efficiency
        / 1000
    )
    return (hover_kw + potential_kw) * climb_time_s / 3600


def mission_energy_for_mass(max_gross_mass_kg: float,
                            mission: MissionProfile,
                            cal: Calibration,
                            a: DesignAssumptions,
                            propeller_count: int = 8,
                            coaxial: bool = True) -> dict:
    if mission.max_payload_kg <= 0:
        raise ValueError("Mission must contain a positive payload.")
    if max_gross_mass_kg <= mission.max_payload_kg:
        raise ValueError("Gross mass must exceed payload mass.")

    geom = vehicle_geometry(
        max_gross_mass_kg, cal, a, propeller_count, coaxial
    )
    rho = air_density(a.site_altitude_m)

    non_payload_mass = max_gross_mass_kg - mission.max_payload_kg
    outbound_mass = non_payload_mass + mission.outbound_payload_kg
    inbound_mass = non_payload_mass + mission.inbound_payload_kg
    scene_mass = non_payload_mass + mission.scene_hover_payload_kg

    out_kwh, out_kw = cruise_energy_kwh(
        outbound_mass, mission.radius_km, geom["area_m2"], rho, cal, a
    )
    in_kwh, in_kw = cruise_energy_kwh(
        inbound_mass, mission.radius_km, geom["area_m2"], rho, cal, a
    )

    scene_hover_kw = electrical_hover_power_kw(
        scene_mass, geom["area_m2"], rho, cal
    )
    scene_kwh = scene_hover_kw * a.scene_hover_minutes / 60

    climb_kwh = climb_energy_kwh(
        outbound_mass, geom["area_m2"], rho, cal, a
    )

    total_kwh = out_kwh + in_kwh + scene_kwh + climb_kwh

    return {
        "total_kwh": total_kwh,
        "outbound_kwh": out_kwh,
        "inbound_kwh": in_kwh,
        "scene_hover_kwh": scene_kwh,
        "climb_kwh": climb_kwh,
        "outbound_power_kw": out_kw,
        "inbound_power_kw": in_kw,
        "scene_hover_power_kw": scene_hover_kw,
        "outbound_mass_kg": outbound_mass,
        "inbound_mass_kg": inbound_mass,
        "scene_mass_kg": scene_mass,
        **geom,
    }


# 8. Bracketed sizing solver

def _bisect_root(f: Callable[[float], float],
                 lo: float,
                 hi: float,
                 tol: float = 1e-5,
                 max_iter: int = 200) -> float:
    """Dependency-free bracketed bisection root solver."""
    flo = f(lo)
    fhi = f(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi)):
        raise ValueError("Root bracket contains non-finite residuals.")
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise ValueError("Root is not bracketed.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if not math.isfinite(fmid):
            hi = mid
            continue
        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)


def estimate_design(mission: MissionProfile,
                    cal: Calibration,
                    a: DesignAssumptions,
                    propeller_count: int = 8,
                    coaxial: bool = True,
                    search_upper_mass_kg: float = 5000.0) -> dict:

    if mission.radius_km < 0:
        raise ValueError("Mission radius cannot be negative.")

    max_payload = mission.max_payload_kg
    lower_mass = max_payload + a.fixed_system_mass_kg + 1.0

    if a.usable_battery_fraction <= 0 or not (0 <= a.reserve_fraction < 1):
        raise ValueError("Battery fractions are invalid.")
    dispatchable_fraction = (
        a.usable_battery_fraction * (1 - a.reserve_fraction)
    )

    omi_per_motor_power_ratio = (
        (propeller_count / (propeller_count - 1)) ** 1.5
        if a.require_omi else 1.0
    )
    maneuver_power_ratio = a.maneuver_thrust_margin ** 1.5
    required_motor_ratio = max(omi_per_motor_power_ratio, maneuver_power_ratio)
    design_power_ratio = max(a.motor_power_margin, required_motor_ratio)

    def mass_balance(gross_mass_kg: float, return_details: bool = False):
        mission_eval = mission_energy_for_mass(
            gross_mass_kg, mission, cal, a, propeller_count, coaxial
        )

        battery_capacity_kwh = mission_eval["total_kwh"] / dispatchable_fraction
        battery_mass = (
            battery_capacity_kwh * 1000
            / cal.battery_specific_energy_wh_kg
        )

        structure_mass = structure_mass_kg(gross_mass_kg, cal, a)

        rho = air_density(a.site_altitude_m)
        hover_at_gross_kw = electrical_hover_power_kw(
            gross_mass_kg, mission_eval["area_m2"], rho, cal
        )
        max_design_power_kw = hover_at_gross_kw * design_power_ratio
        propulsion_mass = (
            max_design_power_kw / a.propulsion_power_density_kw_kg
        )

        calculated_gross = (
            max_payload
            + a.fixed_system_mass_kg
            + structure_mass
            + propulsion_mass
            + battery_mass
        )
        residual = calculated_gross - gross_mass_kg

        if return_details:
            return {
                "residual_kg": residual,
                "battery_capacity_kwh": battery_capacity_kwh,
                "battery_mass_kg": battery_mass,
                "structure_mass_kg": structure_mass,
                "propulsion_mass_kg": propulsion_mass,
                "hover_power_at_mtow_kw": hover_at_gross_kw,
                "max_design_power_kw": max_design_power_kw,
                "dispatchable_battery_fraction": dispatchable_fraction,
                "omi_required_per_motor_power_ratio": omi_per_motor_power_ratio,
                "maneuver_required_power_ratio": maneuver_power_ratio,
                "design_power_ratio": design_power_ratio,
                **mission_eval,
            }
        return residual

    grid = np.geomspace(lower_mass, search_upper_mass_kg, 250)
    previous_m = float(grid[0])
    previous_f = mass_balance(previous_m)
    bracket = None

    for m in grid[1:]:
        m = float(m)
        f = mass_balance(m)
        if math.isfinite(previous_f) and math.isfinite(f) and previous_f * f <= 0:
            bracket = (previous_m, m)
            break
        previous_m, previous_f = m, f

    if bracket is None:
        return {
            "converged": False,
            "mission": mission.name,
            "reason": (
                "No mass-balance root was found within the search range; "
                "under these assumptions the mission is outside the modeled "
                "battery-electric design envelope."
            ),
        }

    root = _bisect_root(lambda m: mass_balance(m), bracket[0], bracket[1])
    details = mass_balance(root, return_details=True)

    # Per-motor requirements at MTOW.
    per_motor_hover_thrust_n = root * G / propeller_count
    per_motor_design_thrust_n = (
        per_motor_hover_thrust_n * a.maneuver_thrust_margin
    )
    per_motor_hover_power_kw = details["hover_power_at_mtow_kw"] / propeller_count
    per_motor_design_power_kw = (
        per_motor_hover_power_kw * details["design_power_ratio"]
    )

    part107_threshold_kg = 55 * LB_TO_KG

    return {
        "converged": True,
        "mission": mission.name,
        "mtow_kg": root,
        "max_payload_kg": max_payload,
        "payload_fraction": max_payload / root,
        "battery_fraction": details["battery_mass_kg"] / root,
        "propeller_count": propeller_count,
        "coaxial": coaxial,
        "per_motor_hover_thrust_n": per_motor_hover_thrust_n,
        "per_motor_design_thrust_n": per_motor_design_thrust_n,
        "per_motor_hover_power_kw": per_motor_hover_power_kw,
        "per_motor_design_power_kw": per_motor_design_power_kw,
        "omi_capable_by_power_margin": (
            a.motor_power_margin
            >= details["omi_required_per_motor_power_ratio"]
        ),
        "above_part107_small_uas_mass": root >= part107_threshold_kg,
        "human_scale_payload_case": mission.human_scale_payload_case,
        **details,
    }


# 9. Mission scenarios

def default_missions(radius_km: float = 5.0) -> list[MissionProfile]:
    """Representative cases. Replace payload values with sourced mission data."""
    return [
        MissionProfile(
            name="Medical / communications delivery",
            outbound_payload_kg=20.0,
            inbound_payload_kg=0.0,
            scene_hover_payload_kg=20.0,
            radius_km=radius_km,
        ),
        MissionProfile(
            name="Heavy rescue-equipment delivery",
            outbound_payload_kg=40.0,
            inbound_payload_kg=0.0,
            scene_hover_payload_kg=40.0,
            radius_km=radius_km,
        ),
        MissionProfile(
            name="Human-scale extraction boundary case",
            outbound_payload_kg=15.0,
            inbound_payload_kg=90.0,
            scene_hover_payload_kg=90.0,
            radius_km=radius_km,
            human_scale_payload_case=True,
        ),
    ]


def report_cases(cal: Calibration,
                 a: DesignAssumptions,
                 radius_km: float = 5.0) -> list[dict]:
    print("=" * 78)
    print(f"MISSION CASES -- {radius_km:.1f} km ONE-WAY RADIUS")
    print("=" * 78)

    results = []
    for mission in default_missions(radius_km):
        r = estimate_design(mission, cal, a)
        results.append(r)
        print(f"\n{mission.name}")
        print(
            f"  Payload out / back / scene : "
            f"{mission.outbound_payload_kg:.0f} / "
            f"{mission.inbound_payload_kg:.0f} / "
            f"{mission.scene_hover_payload_kg:.0f} kg"
        )

        if not r["converged"]:
            print(f"  OUTSIDE MODELED ENVELOPE: {r['reason']}")
            continue

        print(f"  Estimated MTOW              : {r['mtow_kg']:8.1f} kg")
        print(f"  Required battery            : {r['battery_mass_kg']:8.1f} kg "
              f"({r['battery_capacity_kwh']:.2f} kWh nameplate)")
        print(f"  Mission energy              : {r['total_kwh']:8.2f} kWh")
        print(f"  Hover power at MTOW         : {r['hover_power_at_mtow_kw']:8.2f} kW")
        print(f"  Structure residual          : {r['structure_mass_kg']:8.1f} kg")
        print(f"  Propulsion-system mass      : {r['propulsion_mass_kg']:8.1f} kg")
        print(f"  Propeller diameter          : {r['diameter_m']:8.2f} m")
        print(f"  Vehicle span                : {r['span_m']:8.2f} m"
              f"  {'[SPAN LIMITED]' if r['span_limited'] else ''}")
        print(f"  Disk loading at MTOW        : "
              f"{r['mtow_kg'] * G / r['area_m2']:8.1f} N/m^2")
        print(f"  Per-motor design thrust     : {r['per_motor_design_thrust_n']:8.1f} N")
        print(f"  Per-motor design power      : {r['per_motor_design_power_kw']:8.2f} kW")
        print(f"  Battery fraction of MTOW    : {100*r['battery_fraction']:8.1f} %")
        print(f"  Payload fraction of MTOW    : {100*r['payload_fraction']:8.1f} %")
        print(f"  OMI power-margin check      : "
              f"{'PASS' if r['omi_capable_by_power_margin'] else 'FAIL'}")
        if r["above_part107_small_uas_mass"]:
            print("  Regulatory note             : exceeds the 55 lb small-UAS mass "
                  "threshold; assess the applicable pathway separately.")
        if r["human_scale_payload_case"]:
            print("  Human-payload note          : technical boundary case only; this "
                  "model does not establish legal or certification feasibility.")
    print()
    return results


# 10. Sensitivity analysis

def sensitivity(cal: Calibration,
                a: DesignAssumptions,
                mission: MissionProfile) -> None:
    baseline = estimate_design(mission, cal, a)
    if not baseline["converged"]:
        print("Baseline is outside the modeled envelope; sensitivity skipped.\n")
        return

    base_mtow = baseline["mtow_kg"]
    print("=" * 78)
    print(f"SENSITIVITY -- {mission.name}")
    print(f"Baseline MTOW = {base_mtow:.1f} kg")
    print("=" * 78)

    sweeps = [
        ("battery specific energy (Wh/kg)", "battery", [150, 176.4, 200, 250, 300]),
        ("reserve fraction", "reserve_fraction", [0.10, 0.20, 0.30, 0.40]),
        ("site altitude (m)", "site_altitude_m", [0, 1000, 2000, 3000]),
        ("structure mass exponent", "structure_mass_exponent", [1.00, 1.05, 1.10, 1.15, 1.20]),
        ("rotor diameter scaling exponent", "rotor_diameter_scaling_exponent", [0.25, 1/3, 0.40]),
        ("maximum vehicle span (m)", "max_vehicle_span_m", [3.0, 4.0, 5.0, 6.0]),
        ("propulsion power density (kW/kg)", "propulsion_power_density_kw_kg", [2.0, 2.5, 3.0, 4.0, 5.0]),
        ("fixed systems mass (kg)", "fixed_system_mass_kg", [3.0, 5.0, 8.0, 12.0]),
        ("usable battery fraction", "usable_battery_fraction", [0.80, 0.85, 0.90, 0.95]),
        ("scene hover (min)", "scene_hover_minutes", [0, 2, 5, 10]),
    ]

    for label, field, values in sweeps:
        print(f"\n  {label}")
        for value in values:
            if field == "battery":
                trial_cal = replace(
                    cal, battery_specific_energy_wh_kg=float(value)
                )
                trial_a = a
            else:
                trial_cal = cal
                trial_a = replace(a, **{field: float(value)})

            r = estimate_design(mission, trial_cal, trial_a)
            if r["converged"]:
                delta = 100 * (r["mtow_kg"] - base_mtow) / base_mtow
                print(f"    {value:>8.3g} -> {r['mtow_kg']:8.1f} kg  ({delta:+6.1f} %)")
            else:
                print(f"    {value:>8.3g} -> OUTSIDE MODELED ENVELOPE")
    print()


# 11. Design-space plot

def plot_delivery_design_space(cal: Calibration,
                               a: DesignAssumptions,
                               filename: str = "design_space_v2.png") -> None:
    """Plot a delivery mission: payload out, zero payload back."""
    payloads = np.arange(10, 101, 5)
    radii = np.arange(1, 21, 1)
    grid = np.full((len(payloads), len(radii)), np.nan)

    for i, payload in enumerate(payloads):
        for j, radius in enumerate(radii):
            mission = MissionProfile(
                name="Delivery design-space point",
                outbound_payload_kg=float(payload),
                inbound_payload_kg=0.0,
                scene_hover_payload_kg=float(payload),
                radius_km=float(radius),
            )
            r = estimate_design(mission, cal, a)
            if r["converged"]:
                grid[i, j] = r["mtow_kg"]

    radius_mesh, payload_mesh = np.meshgrid(radii, payloads)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    finite = np.isfinite(grid)
    if not finite.any():
        raise RuntimeError("No feasible design-space points were found.")

    filled = ax.contourf(radius_mesh, payload_mesh, grid, levels=18)
    lines = ax.contour(
        radius_mesh,
        payload_mesh,
        grid,
        levels=[100, 200, 400, 800, 1600],
        linewidths=1.0,
    )
    ax.clabel(lines, inline=True, fontsize=8, fmt="%.0f kg")
    fig.colorbar(filled, ax=ax, label="Estimated MTOW (kg)")

    ax.set_xlabel("One-way mission radius (km)")
    ax.set_ylabel("Outbound payload (kg)")
    ax.set_title(
        "Heavy-Lift Delivery Design Space\n"
        "payload outbound, return without payload"
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Design-space contour written to {filename}\n")


# 12. Entry point

def main() -> None:
    ref = ReferenceAircraft()
    assumptions = DesignAssumptions()

    cal = calibrate(ref, assumptions)
    cross_validate_reference(ref, cal)

    report_cases(cal, assumptions, radius_km=5.0)

    sensitivity(
        cal,
        assumptions,
        MissionProfile(
            name="40 kg rescue-equipment delivery, 5 km radius",
            outbound_payload_kg=40.0,
            inbound_payload_kg=0.0,
            scene_hover_payload_kg=40.0,
            radius_km=5.0,
        ),
    )

    plot_delivery_design_space(cal, assumptions)


if __name__ == "__main__":
    main()
