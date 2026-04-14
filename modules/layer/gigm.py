import random
from typing import Dict, List, Sequence

import torch
import torch.nn as nn


class GIGM(nn.Module):
    """
    Practical implementation of the Graph Incentive Generation Model (GIGM).

    This module follows the paper's high-level design:
      1) Intrinsic incentive: collaborative-density Myerson-like credit
         assignment on each agent's first-person ally graph.
      2) Extrinsic incentive: KL divergence between consecutive enemy-graph
         embeddings.

    Notes:
      - The original paper describes a trainable UQVM hypernetwork. In the
        current codebase there is no direct supervised signal for that module.
        To keep the implementation stable and usable, the connected-coalition
        utility is estimated directly from the selected per-agent Q-values.
      - The Shapley/Myerson term is approximated with Monte-Carlo permutation
        sampling, which is consistent with the paper's complexity discussion.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.alpha_int = float(getattr(args, "gigm_alpha_int", 2.0))
        self.alpha_ext = float(getattr(args, "gigm_alpha_ext", 1.0))
        self.reward_scale = float(getattr(args, "gigm_reward_scale", 0.05))
        self.num_samples = int(getattr(args, "gigm_num_samples", 10))
        self.exp_clip = float(getattr(args, "gigm_exp_clip", 5.0))
        self.kl_eps = float(getattr(args, "gigm_kl_eps", 1e-8))

    def forward(self, batch, chosen_action_qvals, mask=None):
        """
        Args:
            batch: EpisodeBatch.
            chosen_action_qvals: [B, T, N] selected Q-values for executed actions.
            mask: optional [B, T, 1] valid-transition mask.

        Returns:
            agent_incentives: [B, T, N]
            team_incentive:   [B, T, 1]
            aux: dict for logging/debugging
        """
        ally_nodes = batch["all_node"][:, :-1].float()
        ally_adj = batch["all_adj"][:, :-1].float()
        enemy_nodes_t = batch["ene_node"][:, :-1].float()
        enemy_adj_t = batch["ene_adj"][:, :-1].float()
        enemy_nodes_tp1 = batch["ene_node"][:, 1:].float()
        enemy_adj_tp1 = batch["ene_adj"][:, 1:].float()

        intrinsic = self._intrinsic_incentive(ally_nodes, ally_adj, chosen_action_qvals.detach())
        extrinsic = self._extrinsic_incentive(enemy_nodes_t, enemy_adj_t, enemy_nodes_tp1, enemy_adj_tp1)

        incentives = self.alpha_int * intrinsic + self.alpha_ext * extrinsic
        incentives = incentives * self.reward_scale

        if mask is not None:
            incentives = incentives * mask.expand_as(incentives)

        team_incentive = incentives.sum(dim=-1, keepdim=True)
        aux = {
            "intrinsic": intrinsic.detach(),
            "extrinsic": extrinsic.detach(),
            "total": incentives.detach(),
        }
        return incentives, team_incentive, aux

    def _intrinsic_incentive(self, ally_nodes, ally_adj, chosen_q):
        # ally_nodes: [B, T, A, N, F], ally_adj: [B, T, A, N, N], chosen_q: [B, T, N]
        B, T, A, N, _ = ally_nodes.shape
        device = chosen_q.device
        mu = torch.zeros(B, T, A, device=device, dtype=chosen_q.dtype)
        alive = (ally_nodes[..., 0].amax(dim=-1) > 0).float()

        for b in range(B):
            for t in range(T):
                q_bt = chosen_q[b, t]
                for i in range(A):
                    if alive[b, t, i].item() == 0:
                        continue
                    nodes_i = ally_nodes[b, t, i]
                    adj_i = ally_adj[b, t, i]
                    mu[b, t, i] = self._estimate_myerson_for_agent(i, nodes_i, adj_i, q_bt)

        denom = alive.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_mu = (mu * alive).sum(dim=-1, keepdim=True) / denom
        return torch.exp((mu - mean_mu).clamp(min=-self.exp_clip, max=self.exp_clip)) * alive

    def _estimate_myerson_for_agent(self, agent_id: int, node_feats, adj, qvals):
        visible = (node_feats[:, 0] > 0)
        visible_ids = torch.nonzero(visible, as_tuple=False).squeeze(-1).tolist()
        if agent_id not in visible_ids:
            return qvals.new_zeros(())
        if len(visible_ids) == 1:
            return qvals[agent_id]

        contrib = qvals.new_zeros(())
        for _ in range(self.num_samples):
            perm = visible_ids[:]
            random.shuffle(perm)
            pos = perm.index(agent_id)
            prev_ids = perm[:pos]
            with_i = prev_ids + [agent_id]
            contrib = contrib + self._restricted_value(with_i, adj, qvals) - self._restricted_value(prev_ids, adj, qvals)
        return contrib / float(self.num_samples)

    def _restricted_value(self, coalition_ids: Sequence[int], full_adj, qvals):
        if len(coalition_ids) == 0:
            return qvals.new_zeros(())

        remaining = set(int(i) for i in coalition_ids)
        total = qvals.new_zeros(())
        while remaining:
            seed = next(iter(remaining))
            comp = self._connected_component(seed, remaining, full_adj)
            for idx in comp:
                remaining.discard(idx)
            comp_ids = sorted(comp)
            comp_adj = full_adj[comp_ids][:, comp_ids]
            density = self._collaborative_density(comp_adj)
            utility = qvals[comp_ids].sum()
            total = total + density * utility
        return total

    def _connected_component(self, start: int, valid_ids: set, full_adj):
        stack = [start]
        visited = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            neigh = torch.nonzero(full_adj[cur] > 0, as_tuple=False).squeeze(-1).tolist()
            for nxt in neigh:
                if int(nxt) in valid_ids and int(nxt) not in visited:
                    stack.append(int(nxt))
        return visited

    def _collaborative_density(self, adj):
        n = adj.shape[0]
        if n <= 1:
            return adj.new_tensor(1.0)
        num_edges = adj.sum() / 2.0
        return (2.0 * num_edges) / float(n * (n - 1))

    def _extrinsic_incentive(self, node_t, adj_t, node_tp1, adj_tp1):
        emb_t = self._enemy_graph_embedding(node_t, adj_t)
        emb_tp1 = self._enemy_graph_embedding(node_tp1, adj_tp1)

        p = torch.softmax(emb_tp1, dim=-1)
        q = torch.softmax(emb_t, dim=-1)
        kl = (p * ((p + self.kl_eps).log() - (q + self.kl_eps).log())).sum(dim=-1)

        alive = (node_t[..., 0].amax(dim=-1) > 0).float()
        return kl * alive

    def _enemy_graph_embedding(self, node, adj):
        # node: [B, T, A, E, F], adj: [B, T, A, E, E]
        mask = node[..., 0:1]
        count = mask.sum(dim=-2).clamp(min=1.0)
        mean_feat = (node * mask).sum(dim=-2) / count

        degree = adj.sum(dim=-1, keepdim=True)
        if adj.shape[-1] > 1:
            degree = degree / float(adj.shape[-1] - 1)
        mean_degree = (degree * mask).sum(dim=-2) / count
        var_degree = (((degree - mean_degree.unsqueeze(-2)) ** 2) * mask).sum(dim=-2) / count
        return torch.cat([mean_feat, mean_degree, var_degree], dim=-1)
