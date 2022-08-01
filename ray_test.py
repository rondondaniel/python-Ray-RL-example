# Let's code our multi-agent environment.

import gym
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
import time
import pprint
import ray

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Import a Trainable (one of RLlib's built-in algorithms):
# We use the PPO algorithm here b/c its very flexible wrt its supported
# action spaces and model types and b/c it learns well almost any problem.
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import PPOTFPolicy

class MultiAgentArena(MultiAgentEnv):
    def __init__(self, config=None):
        """ Config takes in width, height, and ts """
        config = config or {}
        # Dimensions of the grid.
        self.width = config.get("width", 10)
        self.height = config.get("height", 10)

        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 100)

        self.observation_space = MultiDiscrete([self.width * self.height,
                                                self.width * self.height])
        # 0=up, 1=right, 2=down, 3=left.
        self.action_space = Discrete(4)

        # Reset env.
        self.reset()
        
    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.
        self.agent1_pos = [0, 0]  # upper left corner
        self.agent2_pos = [self.height - 1, self.width - 1]  # lower bottom corner

        # Accumulated rewards in this episode.
        self.agent1_R = 0.0
        self.agent2_R = 0.0

        # Reset agent1's visited fields.
        self.agent1_visited_fields = set([tuple(self.agent1_pos)])

        # How many timesteps have we done in this episode.
        self.timesteps = 0

        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.
        
        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """
        
        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit

        # Agent2 always moves first.
        # events = [collision|agent1_new_field]
        events = self._move(self.agent2_pos, action["agent2"], is_agent1=False)
        events = self._move(self.agent1_pos, action["agent1"], is_agent1=True)

        # Useful for rendering.
        self.collision = "collision" in events
            
        # Get observations (based on new agent positions).
        obs = self._get_obs()

        # Determine rewards based on the collected events:
        r1 = -1.0 if "collision" in events else 1.0 if "agent1_new_field" in events else -0.5
        r2 = 1.0 if "collision" in events else -0.1

        self.agent1_R += r1
        self.agent2_R += r2
        
        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }

        return obs, rewards, dones, {}  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs dict (agent name to discrete-pos tuple) using each
        agent's current x/y-positions.
        """
        ag1_discrete_pos = self.agent1_pos[0] * self.width + \
            (self.agent1_pos[1] % self.width)
        ag2_discrete_pos = self.agent2_pos[0] * self.width + \
            (self.agent2_pos[1] % self.width)
        return {
            "agent1": np.array([ag1_discrete_pos, ag2_discrete_pos]),
            "agent2": np.array([ag2_discrete_pos, ag1_discrete_pos]),
        }

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # Solve collisions.
        # Make sure, we don't end up on the other agent's position.
        # If yes, don't move (we are blocked).
        if (is_agent1 and coords == self.agent2_pos) or (not is_agent1 and coords == self.agent1_pos):
            coords[0], coords[1] = orig_coords
            # Agent2 blocked agent1 (agent1 tried to run into agent2)
            # OR Agent2 bumped into agent1 (agent2 tried to run into agent1)
            return {"collision"}

        # No agent blocking -> check walls.
        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.height:
            coords[0] = self.height - 1
        if coords[1] < 0:
            coords[1] = 0
        elif coords[1] >= self.width:
            coords[1] = self.width - 1

        # If agent1 -> "new" if new tile covered.
        if is_agent1 and not tuple(coords) in self.agent1_visited_fields:
            self.agent1_visited_fields.add(tuple(coords))
            return {"agent1_new_field"}
        # No new tile for agent1.
        return set()

    def render(self, mode=None):
        print("_" * (self.width + 2))
        for r in range(self.height):
            print("|", end="")
            for c in range(self.width):
                field = r * self.width + c % self.width
                if self.agent1_pos == [r, c]:
                    print("1", end="")
                elif self.agent2_pos == [r, c]:
                    print("2", end="")
                elif (r, c) in self.agent1_visited_fields:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("|")
        print("‾" * (self.width + 2))
        print(f"{'!!Collision!!' if self.collision else ''}")
        print("R1={: .1f}".format(self.agent1_R))
        print("R2={: .1f}".format(self.agent2_R))
        print()

class DummyTrainer:
    """Dummy Trainer class used in Exercise #1.

    Use its `compute_action` method to get a new action for one of the agents,
    given the agent's observation (a single discrete value encoding the field
    the agent is currently in).
    """

    def compute_action(self, single_agent_obs=None):
        # Returns a random action for a single agent.
        return np.random.randint(4)  # Discrete(4) -> return rand int between 0 and 3 (incl. 3).

def dummy_test():
    dummy_trainer = DummyTrainer()

    # Check, whether it's working.
    for _ in range(3):
        # Get action for agent1 (providing agent1's and agent2's positions).
        print("action_agent1={}".format(dummy_trainer.compute_action(np.array([0, 99]))))
    
        # Get action for agent2 (providing agent2's and agent1's positions).
        print("action_agent2={}".format(dummy_trainer.compute_action(np.array([99, 0]))))
    
        print()

def first_test():
    print("Launch First Test")
    env = MultiAgentArena()
    
    obs = env.reset()
    
    # Agent1 will move down, Agent2 moves up.
    obs, rewards, dones, infos = env.step(action={"agent1": 2, "agent2": 0})
    
    env.render()
    
    print("Agent1's x/y position={}".format(env.agent1_pos))
    print("Agent2's x/y position={}".format(env.agent2_pos))
    print("Env timesteps={}".format(env.timesteps))

def by_timestep_test():
    print("Launch By Timestep Test")
    env = MultiAgentArena()
    dummy_trainer = DummyTrainer()
    
    # Start coding here inside this `with`-block:
    # 1) Reset the env.
    obs = env.reset()
    
    # 2) Enter an infinite while loop (to step through the episode).
    while True:

        # 3) Calculate both agents' actions individually, using dummy_trainer.compute_action([individual agent's obs])
        action_agent1 = dummy_trainer.compute_action(np.array([0, 99]))
        action_agent2 = dummy_trainer.compute_action(np.array([99, 0]))

        # 4) Compile the actions dict from both individual agents' actions.
        action_dict = {"agent1": action_agent1, "agent2": action_agent2}

        # 5) Send the actions dict to the env's `step()` method to receive: obs, rewards, dones, info dicts
        obs, rewards, dones, infos = env.step(action=action_dict)

        # 6) We'll do this together: Render the env.
        # Don't write any code here (skip directly to 7).
        time.sleep(0.08)
        env.render()

        # 7) Check, whether the episde is done, if yes, break out of the while loop.
        if dones["__all__"]:
            break

def train_test():
    # Start a new instance of Ray (when running this tutorial locally) or
    # connect to an already running one (when running this tutorial through Anyscale).
    # ray.shutdown()
    ray.init()  # Hear the engine humming? ;)

    # Specify a very simple config, defining our environment and some environment
    # options (see environment.py).
    config = {
        "env": MultiAgentArena,
        "env_config": {
            "config": {
                "width": 10,
                "height": 10,
                "ts": 100,
            },
        },
    
        # !PyTorch users!
        "framework": "tf",  # If users have chosen to install torch instead of tf.
    
        "create_env_on_driver": True,
    }

    # Instantiate the Trainer object using above config.
    rllib_trainer = PPOTrainer(config=config)

    # Runs 1 Iteration of Training
    results = rllib_trainer.train()

    # Delete the config from the results for clarity.
    # Only the stats will remain, then.
    del results["config"]
    # Pretty print the stats.
    pprint.pprint(results)
    del rllib_trainer

    # Run this if neccessary
    ray.shutdown()

def train_with_policies():
    # Init
    print("Launch a Training with Policies")
    env = MultiAgentArena()
    ray.init()  # Hear the engine humming? ;)

    # Exercise 2
    # 1) Define the policies definition dict:
    # Each policy in there is defined by its ID (key) mapping to a 4-tuple (value):
    # - Policy class (None for using the "default" class, e.g. PPOTFPolicy for PPO+tf or PPOTorchPolicy for PPO+torch).
    # - obs-space (we get this directly from our already created env object).
    # - act-space (we get this directly from our already created env object).
    # - config-overrides dict (leave empty for using the Trainer's config as-is)
    policies = {
        ### Modify Code here ####
        "policy1": (PPOTFPolicy, env.observation_space, env.action_space, {}),
        "policy2": (PPOTFPolicy, env.observation_space, env.action_space, {}),
    }
    # Note that now we won't have a "default_policy" anymore, just "policy1" and "policy2".

    # 2) Defines an agent->policy mapping function.
    # The mapping here is M (agents) -> N (policies), where M >= N.
    def policy_mapping_fn(agent_id: str) -> str:
        # Make sure agent ID is valid.
        assert agent_id in ["agent1", "agent2"], f"ERROR: invalid agent ID {agent_id}!"
        ### Modify Code here ####
        if agent_id == "agent2":
            return "policy2"
        else:
            return "policy1"

        return None

    config = {
        "env": MultiAgentArena,  # "my_env" <- if we previously have registered the env with `tune.register_env("[name]", lambda config: [returns env object])`.
        "env_config": {
            "config": {
                "width": 10,
                "height": 10,
                "ts": 100,
            },
        },
        # !PyTorch users!
        "framework": "tf",  # If users have chosen to install torch instead of tf.
        "create_env_on_driver": True,
    }

    # 3) Adding the above to our config.
    ### Modify Code here ####
    config.update({
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
    })

    pprint.pprint(config)
    print()
    print(f"agent1 is now mapped to {policy_mapping_fn('agent1')}")
    print(f"agent2 is now mapped to {policy_mapping_fn('agent2')}")

    rllib_trainer = PPOTrainer(config=config)

    # 4) Run `train()` n times. Repeatedly call `train()` now to see rewards increase.
    # Move on once you see (agent1 + agent2) episode rewards of 10.0 or more.
    for _ in range(10):
        ### Modify Code here ####
        results = rllib_trainer.train()
        r1 = results['policy_reward_mean']['policy1']
        r2 = results['policy_reward_mean']['policy2']
        r = r1 + r2
        print(f"Training Iteration={rllib_trainer.iteration}: R(\"return\")={r} R1={r1} R2={r2}")

    ray.shutdown()

    return rllib_trainer

def evaluate(rllib_trainer):
    ray.init()
    env = MultiAgentArena()
    obs = env.reset()

    while True:
        a1 = rllib_trainer.compute_action(obs["agent1"], policy_id="policy1")
        a2 = rllib_trainer.compute_action(obs["agent2"], policy_id="policy2")    
        obs, rewards, dones, _ = env.step({"agent1": a1, "agent2": a2})
        time.sleep(0.08)
        env.render()
        if dones["agent1"]:
          break
    
    ray.shutdown()

# In case you encounter the following error during our tutorial: `RuntimeError: Maybe you called ray.init twice by accident?`
# Try: `ray.shutdown() + ray.init()` or `ray.init(ignore_reinit_error=True)`
if __name__ == "__main__":
    #first_test()
    #dummy_test()
    #by_timestep_test()
    #train_test()
    trainer = train_with_policies()
    evaluate(trainer)
