# ComplexBench-Previs24 Task Index

AI-assisted original brief-only pilot; no reference media or measured results.

| ID | Family | Split | Title |
| --- | --- | --- | --- |
| cvb-previs-gallery-doorway-orbit | camera_choreography | train | Doorway entry and sculpture orbit |
| cvb-previs-model-street-crane | camera_choreography | validation | Street-level to rooftop reveal |
| cvb-previs-desk-object-parallax | camera_choreography | test | Foreground parallax and terminal overhead view |
| cvb-previs-porter-pillar-luggage | occlusion_reentry | train | Porter reappears with unchanged luggage |
| cvb-previs-conveyor-tunnel-marked-crate | occlusion_reentry | validation | Marked crate through an opaque conveyor tunnel |
| cvb-previs-cart-behind-divider | occlusion_reentry | test | Loaded cart behind a room divider |
| cvb-previs-workshop-three-routes | multi_actor_blocking | train | Three workers around one fixed workbench |
| cvb-previs-dance-crossing-lanes | multi_actor_blocking | validation | Depth-separated crossing and synchronized stop |
| cvb-previs-cafe-yield-at-door | multi_actor_blocking | test | Yielding at a narrow doorway |
| cvb-previs-cup-handover | contact_and_transfer | train | Visible two-person cup handover |
| cvb-previs-magnet-lift-and-release | contact_and_transfer | validation | Magnetic pickup with a controlled release |
| cvb-previs-book-shelf-support | contact_and_transfer | test | Two-handed book placement onto a shelf |
| cvb-previs-domino-ramp-bell | causal_mechanisms | train | Domino trigger releases a ramp ball |
| cvb-previs-lever-lifts-barrier | causal_mechanisms | validation | Lever opens a barrier before a trolley moves |
| cvb-previs-waterwheel-drives-flag | causal_mechanisms | test | Water flow starts a visible geared mechanism |
| cvb-previs-workshop-three-views | multishot_spatial_continuity | train | One workshop action across three viewpoints |
| cvb-previs-platform-bench-orientation | multishot_spatial_continuity | validation | Platform geography across reverse views |
| cvb-previs-kitchen-open-drawer | multishot_spatial_continuity | test | Persistent drawer state across camera cuts |
| cvb-previs-courtyard-three-mediums | geometry_style_disentanglement | train | Fixed courtyard rendered in three visual mediums |
| cvb-previs-chair-material-variants | geometry_style_disentanglement | validation | One chair design in three materials |
| cvb-previs-train-set-style-triptych | geometry_style_disentanglement | test | Train-set layout through three illustration styles |
| cvb-previs-footstep-spotlight-follow | event_lighting_synchronization | train | Spotlight follows successive stage positions |
| cvb-previs-door-light-spill | event_lighting_synchronization | validation | Door opening controls a wedge of light |
| cvb-previs-swinging-lamp-shadow | event_lighting_synchronization | test | Moving lamp produces a corresponding cast shadow |

Each task lasts 12 seconds. The smoke6 subset contains the first two families.
See ../../docs_previs_graph_extension.md for protocol and implementation boundaries.
