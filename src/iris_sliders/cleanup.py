"""Best-effort cleanup of owned hooks and borrowed native flags.

Every action is attempted. A primary work failure remains the raised exception;
cleanup failures are attached as notes rather than replacing it.
"""
import sys


def run_cleanup(actions, *, primary=None):
    primary = sys.exception() if primary is None else primary
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
    def restore():
        actions = []
        for module, mode, attributes in modules:
            for key in ('gradient_checkpointing', '_gradient_checkpointing_func'):
                actions.append(lambda m=module, k=key, a=attributes: restore_attribute(m, k, a))
            actions.append(lambda m=module, value=mode: setattr(m, 'training', value))
        actions.extend(lambda p=p, flag=flag: p.requires_grad_(flag) for p, flag in gradients)
        run_cleanup(actions, primary=sys.exception())
    return restore
