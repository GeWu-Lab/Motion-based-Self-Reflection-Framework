import torch
import torch.nn as nn

REGISTERED_POLICIES = {}

def register_policy(policy_class):
    """Register a policy class with the registry."""
    policy_name = policy_class.__name__.lower()
    if policy_name in REGISTERED_POLICIES:
        raise ValueError("Cannot register duplicate policy ({})".format(policy_name))

    REGISTERED_POLICIES[policy_name] = policy_class


def get_policy_class(policy_name):
    """Get the policy class from the registry."""
    if policy_name.lower() not in REGISTERED_POLICIES:
        raise ValueError(
            "Policy class with name {} not found in registry".format(policy_name)
        )
    return REGISTERED_POLICIES[policy_name.lower()]


def get_policy_list():
    return REGISTERED_POLICIES


class PolicyMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all policies that should not be registered here.
        _unregistered_policies = ["BasePolicy"]

        if cls.__name__ not in _unregistered_policies:
            register_policy(cls)
        return cls
    

class BasePolicy(nn.Module, metaclass=PolicyMeta):
    def __init__(self) :
        super().__init__()
        
    def get_action(self, obs):
        raise NotImplementedError
    
    def compute_loss(self, obs, gt_action):
        raise NotImplementedError

