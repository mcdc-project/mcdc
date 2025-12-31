import mcdc.code_factory.adapt as adapt


@adapt.toggle("sensitivity")
def score_resp_cum_tracklength(particle_container, distance, mcdc, data):
    """
    Sensitivity-only track-length scoring into particle["resp_cum"][k],
    where k indexes the user-defined response regions.

    Current definition: response regions are specified by cell_ID membership.
    """
    # Keep consistent with other tallying behavior
    if not mcdc["cycle_active"]:
        return

    P = particle_container[0]
    cell_id = P["cell_ID"]

    resp_cells = mcdc["settings"]["sensitivity_resp_cell_IDs"]
    # Find which response region this cell corresponds to (small N_resp => linear scan OK)
    for k in range(resp_cells.shape[0]):
        if cell_id == resp_cells[k]:
            P["resp_cum"][k] += P["w"] * distance
            break
