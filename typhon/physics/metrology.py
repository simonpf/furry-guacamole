"""Functions related to metrology

All functions in this module need sympy.
"""

import warnings

def express_uncertainty(expr, aliases={}, on_failure="raise",
        collect_failures=None, return_sensitivities=False,
        return_components=False):
    """For a sympy expression, calculate uncertainty.

    Uncertainty is given in the Guides to Uncertainties and Measurements
    (GUM), see http://www.bipm.org/en/publications/guides/gum.html
    equations 10 and 13.

    This takes all free symbols.  If you have IndexedBase symbols, you may
    want to pass them into aliases {free_symbol:
    indexed_base_symbol[index]} is this is not identified automatically.

    Limitation: currently assumes all arguments of the expression have
    uncorrelated errors!

    Arguments:
        
        expr (Expr): Expression for which to calculate uncertainty
        aliases (Mapping): Mapping of replacements to apply prior.
        on_failure (str): Signals what to do when some variables cannot be
            differentiated against.  This appears to be the case for
            indexed quantities (see
            https://github.com/sympy/sympy/issues/12191).  Default is
            'raise', but can set to 'warn' instead.
        collect_failures (set): Pass in a set object where failures will
            be collected, assuming on_failure is 'warn'.
        return_sensitivities (boolean): If True, return dictionary with
            sensitivity coefficients (as expressions).  Defaults to False.
        return_components (boolean): If True, return dictionary with
            component of evaluated uncertainty per argument/effect.
            Only considers uncorrelated part (which is the only thing
            implemented so far anyway).

    Returns:
        
        Expression indicating uncertainty
    """

    if collect_failures is None:
        collect_failures = set()
    import sympy
    u = sympy.Function("u")
    rv = sympy.sympify(0)
    sensitivities = {}
    components = {}
    for sym in recursive_args(expr):
        sym = aliases.get(sym, sym)
        try:
            sigma = sympy.diff(expr, sym)
            comp = sigma**2 * u(sym)**2
            rv += comp 
            sensitivities[sym] = sigma
            components[sym] = sympy.sqrt(comp)
        except ValueError as v:
            if on_failure == "raise" or not "derivative" in v.args[0]:
                raise
            else:
                warnings.warn("Unable to complete derivative on {!s}. "
                    "IGNORING uncertainty on {!s}!  Derivative failed with: "
                    "{:s}".format(expr, sym, v.args[0]))
                collect_failures.add(sym)
    unc = sympy.sqrt(rv)
    if return_sensitivities and return_components:
        return (unc, sensitivities, components)
    elif return_sensitivities:
        return (unc, sensitivities)
    elif return_components:
        return (unc, components)
    else:
        return unc

def recursive_args(expr, stop_at=None, partial_at=None):
    """Get arguments for `expr`, stopping at certain types

    Get all arguments in expression down to the levels in `expr`.  When
    `expr` is only `{sympy.Symbol}` this is identical to
    `expr.free_symbols`, but in some cases we want to retain IndexedBase,
    for example, when evaluating uncertainies.

    This is mainly a helper for express_uncertainty where we don't want to
    descend beyond Indexed quantities
    """

    import sympy
    import sympy.concrete.expr_with_limits
    if stop_at is None:
        stop_at = (sympy.Symbol, sympy.Indexed)
    if partial_at is None:
        partial_at = sympy.concrete.expr_with_limits.ExprWithLimits
    if isinstance(expr, stop_at):
        # to return {expr} here leads to infinite recursion!
        return set()
    if isinstance(expr, partial_at):
        return recursive_args(expr.args[0])
    args = set()
    for arg in expr.args:
        if isinstance(arg, stop_at):
            args.add(arg)
        elif isinstance(arg, partial_at):
            # in the case arg.args[0] is a stop_at, recursive_args will
            # return empty; but then we still need to add the expression a
            # level higher up.  For example, support we have
            # arg=Sum(T_PRT[n], (n, 0, N)), then arg.args[0]=T_PRT[n],
            # recursive_args(arg.args[0]) = set() (if sympy.Indexed is in
            # stop_at, such as by default), and neither gets added.
            args.update((
                {arg.args[0]} if isinstance(arg.args[0], stop_at) else set()) |
                recursive_args(arg.args[0], stop_at=stop_at))
        else:
            args.update(recursive_args(arg, stop_at=stop_at))
    return args
