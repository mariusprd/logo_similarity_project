# src/modules/grouping.py (Adapted for DB)

import networkx as nx

class Grouper:
    def __init__(self, db_client):
        """
        Initialize the Grouper with a database client.
        """
        self.db = db_client.logo_db

    def group_by_graph_from_db(self, similarity_threshold=0.9):
        """
        Performs graph-based grouping using similarity data fetched directly from MongoDB.

        Args:
            similarity_threshold (float): The minimum phash_similarity to consider two logos linked.

        Returns:
            A dictionary mapping group IDs to lists of logo domains.
        """
        # Fetch all pairs of logos that meet the similarity threshold
        # We also fetch the domains associated with the logo IDs for the final output.
        pipeline = [
            {"$match": {"phash_similarity": {"$gte": similarity_threshold}}},
            {"$lookup": {"from": "logos", "localField": "logo_id_1", "foreignField": "_id", "as": "logo1_info"}},
            {"$lookup": {"from": "logos", "localField": "logo_id_2", "foreignField": "_id", "as": "logo2_info"}},
            {"$unwind": "$logo1_info"},
            {"$unwind": "$logo2_info"},
        ]
        similar_pairs = list(self.db.similarities.aggregate(pipeline))
        
        if not similar_pairs:
            return {}

        G = nx.Graph()
        
        # We need a map of all unique domains involved
        all_domains = set()

        for pair in similar_pairs:
            domain1 = pair['logo1_info']['domain']
            domain2 = pair['logo2_info']['domain']
            all_domains.add(domain1)
            all_domains.add(domain2)
            G.add_edge(domain1, domain2)

        # Find connected components (each component is a group)
        connected_components = list(nx.connected_components(G))

        # Format the output
        groups = {f"group_{i}": list(component) for i, component in enumerate(connected_components)}
        
        # Add single-item groups for domains that had no similar pairs
        grouped_domains = {item for sublist in groups.values() for item in sublist}
        ungrouped_domains = all_domains - grouped_domains
        
        next_group_idx = len(groups)
        for i, domain in enumerate(ungrouped_domains):
            groups[f"group_{next_group_idx + i}"] = [domain]

        return groups