# Authors

This repository is a fork of the 2025 Carleton College Senior Capstone
project on TurtleBot4 obstacle avoidance (Fall 2025 -- Winter 2026),
with subsequent modifications and new packages added by Daniel Scheider
in 2026.

The original Capstone project was completed under the supervision of
Prof. Chelsey Edge, with additional technical support from Mike Tie.

## Original Capstone Team (2025 -- 2026)

The following contributors authored the original A*, D* Lite, JPS, and
DWA planner implementations and supporting infrastructure that this fork
builds on. Listed alphabetically, with the areas each person worked on:

- Daniel Scheider -- A*, DWA, stacked functionality
- Devin Dennis -- A*, D* Lite
- Dexter Kong -- DWA costmap grid (since reimplemented as a standalone
  obstacle-grid node with different logic)
- Kat Smiricinschi -- A*, D* Lite
- Oliver Black-Johnston -- JPS, stacked functionality
- Raquel Emeka -- JPS

See the upstream repository's git history for line-level attribution.

## This Fork (2026)

All work in this repository after the Capstone concluded was done by
Daniel Scheider. This includes the new `nav_interfaces`, `obstacle_grid`,
and `nav_server` packages, the refactor and style cleanup of the original
planner packages, LIDAR raycasting and the shared obstacle grid, and the
unified Navigate action server and CLI client.

## File-Level Headers

Files that have been substantially rewritten (rather than incrementally
edited) carry a short header indicating their original-and-rewrite
authorship. Files modified surgically retain their git-blame attribution
without an in-file header.
