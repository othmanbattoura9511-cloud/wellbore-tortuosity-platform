project:
  name: Wellbore Tortuosity Analytics Platform
  unit_system: metric
survey:
  md_col: MD
  inc_col: Inclination
  azi_col: Azimuth
  dls_col: DLS
  default_dls_unit_length: 30
  dense_spacing_threshold_m: 5
  coarse_spacing_threshold_m: 30
classification:
  vertical_inc_max: 10
  lateral_inc_min: 80
simulation:
  friction_coefficient: 0.25
  torque_factor: 1.0
  drag_factor: 1.0
  buckling_sensitivity: 1.0
reporting:
  output_dir: reports
