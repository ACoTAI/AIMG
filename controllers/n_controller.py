from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)

        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def _build_graph_inputs(self, batch, t):

        all_node = batch["all_node"][:, t]
        all_adj = batch["all_adj"][:, t]
        ene_node = batch["ene_node"][:, t]
        ene_adj = batch["ene_adj"][:, t]

        return (all_node.reshape(-1, self.args.n_agents, all_node.shape[-1]), all_adj.reshape(-1, self.args.n_agents, self.args.n_agents)),\
               (ene_node.reshape(-1, self.args.n_enemies, ene_node.shape[-1]), ene_adj.reshape(-1, self.args.n_enemies, self.args.n_enemies))

    def todata(self, graph):
        all_node = graph[0].reshape(-1, self.args.n_agents, graph[0].shape[-1])
        all_edge = graph[1].reshape(-1, self.args.n_agents, self.args.n_agents)
        ene_node = graph[2].reshape(-1, self.args.n_enemies, graph[2].shape[-1])
        ene_edge = graph[3].reshape(-1, self.args.n_enemies, self.args.n_enemies)


        return (all_node, all_edge), (ene_node, ene_edge)


    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
            try:
                self.agent.training = False
            except Exception as e:
                pass
            
        agent_inputs = self._build_inputs(ep_batch, t)
        # avail_actions = ep_batch["avail_actions"][:, t]
        graph_inputs = self._build_graph_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, graph_inputs, self.hidden_states)
        try:
            # self.agent.train()
            self.agent.training = False
        except Exception as e:
            pass

        return agent_outs