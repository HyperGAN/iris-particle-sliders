"""Explicit-primary cleanup of owned hooks and borrowed native flags.

Every action is attempted. Only the operation's explicitly supplied failure is
preserved; an unrelated surrounding except block cannot suppress cleanup errors.
Completed actions are released so repeated close cannot replay stale snapshots.
"""


def run_cleanup(actions, *, primary=None):
    failures = []
    for action in actions:
        try:
            action()
        except BaseException as error:
            failures.append(error)
    if failures:
        error = primary if primary is not None else failures[0]
        for failure in failures:
            error.add_note('Iris cleanup failure: ' + repr(failure))
        if primary is None:
            raise error


class Cleanup:
    """Own pending actions once; retry failures without replaying successes."""
    def __init__(self, actions):
        self.pending = list(actions)

    def __call__(self):
        actions, self.pending = self.pending, []
        def attempt(action):
            try:
                action()
            except BaseException:
                self.pending.append(action)
                raise
        run_cleanup([lambda action=action: attempt(action) for action in actions])


def native_cleanup(root):
    modules = [(m, m.training, {key: getattr(m, key) for key in
                ('gradient_checkpointing', '_gradient_checkpointing_func') if hasattr(m, key)})
               for m in root.modules()]
    gradients = [(p, p.requires_grad) for p in root.parameters()]
    def restore_attribute(module, key, previous):
        if key in previous:
            setattr(module, key, previous[key])
        elif hasattr(module, key):
            delattr(module, key)
    actions = []
    for module, mode, attributes in modules:
        for key in ('gradient_checkpointing', '_gradient_checkpointing_func'):
            actions.append(lambda m=module, k=key, a=attributes: restore_attribute(m, k, a))
        actions.append(lambda m=module, value=mode: setattr(m, 'training', value))
    actions.extend(lambda p=p, flag=flag: p.requires_grad_(flag) for p, flag in gradients)
    return Cleanup(actions)
