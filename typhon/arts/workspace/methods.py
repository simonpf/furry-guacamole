""" The methods submodule.

This module exposes all ARTS workspace methods represented by WorkspaceMethod object.

The methods are loaded dynamically when the module is imported, which ensures that they
up to date with the current ARTS build.

Attributes:

     workspace_methods(dict): Dictionary containing all ARTS workspace methods.

"""

import ast
import re
import ctypes as c

from typhon.arts.workspace.api       import arts_api, VariableValueStruct
from typhon.arts.workspace.variables import WorkspaceVariable, group_ids, group_names
from typhon.arts.workspace import variables, workspace

class WorkspaceMethod:
    """
    The WorkspaceMethod class represents ARTS workspace methods. Each workspace method
    provided a call function that forwards the call to the function towards the ARTS C API.

    Attributes:
        m_ids([int]):       Indices of supergeneric overloads of this WSM
        name(str):          The name of the method as defined in methods.cc
        description(str):   The documentation of the method as defined in methods.cc
        outs([int]):        Indices of the output variables of the method.
        n_out(int):         The number of output arguments.
        n_g_out(int):       The number of generic outputs.
        g_out([str]):       The names of the generic output arguments.
        g_out_types([dict]): List of dicts associating the name of a generic output
                             with its types for each of the supergeneric overloads.
        n_in(int):          The number of input arguments.
        ins([int]):         The indices of the input arguments of the WSM.
        n_g_in(int):        The number of generic input arguments.
        g_in_types([dict]): List of dicts associating the name of a generic input to the
                            expected type for each supergeneric overload of the method.
        g_in_default(dict)  Dictionary containing the default values for each generic parameter.
        g_in([str]):        List of names of the generic input arguments.
    """

    # Regular expression to that matches <Group>Create WSMs
    create_regexp = re.compile("^(\w*)Create$")

    def __init__(self, m_id, name, description, outs, g_out_types, ins, g_in_types):
        """Create a WorkspaceMethod object from a given id, name, description and types of
        generic input and output arguments.

        Args:
        m_id(int):          The index identifying the method in the C API
        name(str):          The name of the method
        description(str):   Method documentation
        outs([int]):        List of indices of the (non-generic) output arguments of the method.
        g_out_types([int]): Group ids of the generic output types
        outs([int]):        List of indices of the (non-generic) input arguments of the method.
        g_in_types([int]):  Group ids of the generic input types
        """
        self.m_ids       = [m_id]
        self.name        = name
        self.description = description

        # Output
        self.outs  = outs
        self.n_out = len(outs)

        # Generic Output
        self.n_g_out     = len(g_out_types)
        self.g_out_types = [WorkspaceMethod.get_output_dict(m_id, g_out_types)]
        self.g_out       = [k for k in self.g_out_types[0]]

        # Input
        self.ins  = ins
        self.n_in = len(ins)

        # Generic Input
        self.n_g_in       = len(g_in_types)
        self.g_in_types   = [WorkspaceMethod.get_input_dict(m_id, g_in_types)]
        self.g_in_default = WorkspaceMethod.get_default_input_dict(m_id, g_in_types)
        self.g_in         = [k for k in self.g_in_types[0]]

        self.is_create = False
        if (WorkspaceMethod.create_regexp.match(name)):
                self.is_create = True

    def add_overload(self, m_ids, g_in_types, g_out_types):
        """ Add one or more overloads to a workspace method.

        Use this function to add a supergeneric overload to a WorkspaceMethod object
        so that is will be considered in overload resolution when call(...) is called.

        TODO: Simplify this to take a WorkspaceMethod object.

        Args:
            m_ids([int]): The method ids of the supergeneric ARTS WSMs which should be added
                          to the list of overloads.
            g_in_types ([dict]): List of dicts containing the mappings between argument
                                 names and indices of expected groups
            g_out_types ([dict]): List of dicts containing the mappings between argument
                                  names and indices of expected groups
        """
        self.m_ids += m_ids
        self.g_in_types += g_in_types
        self.g_out_types += g_out_types

    @staticmethod
    def get_input_dict(m_id, in_types):
        """Get mapping of names of generic input variables to indices of groups.

        Args:
            m_id(int):       Index of the method.
            in_types([int]): List of group indices of the generic input arguments.
        Return:
            dict: The mapping.
        """
        res = dict()
        for i,t in enumerate(in_types):
            res[arts_api.get_method_g_in(m_id, i).decode("utf8")] = t
        return res

    @staticmethod
    def get_default_input_dict(m_id, g_in):
        """Get dict mapping names of generic input arguments to default values.

        Is None if no default value for a given generic input is given.

        Args:
            m_id(int):   Index of the method.
            g_in([str]): Names of the generic input arguments.
        Return:
            dict: The mapping.
        """
        res = dict()
        for i,t in enumerate(g_in):
            k = arts_api.get_method_g_in(m_id, i).decode("utf8")
            d = arts_api.get_method_g_in_default(m_id, i).decode("utf8")
            if d == "":
                pass
            elif d[0] == "@":
                d = None
            else:
                try:
                    d = WorkspaceVariable.convert(group_names[t], ast.literal_eval(d))
                    res[k] = d
                except:
                    res[k] = d
        return res

    @staticmethod
    def get_output_dict(m_id, out_types):
        """Get mapping of names of generic output variables to indices of groups.

        Args:
            m_id(int):       Index of the method.
            in_types([int]): List of group indices of the generic output arguments.
        Return:
            dict: The mapping.
        """
        res = dict()
        for i,t in enumerate(out_types):
            res[arts_api.get_method_g_out(m_id, i).decode("utf8")] = t
        return res

    def _parse_output_input_lists(self, ws, args, kwargs):
        n_args = self.n_g_out + self.n_g_in

        ins  = self.ins
        temps = []

        # Add positional arguments to kwargs
        if (len(args)) and (len(args)) < self.n_g_out + self.n_in:
            raise Exception("Only " + str(len(args)) + "positional arguments provided " +
                            "but WSM " + self.name + "requires at least "
                            + str(self.n_g_out + self.n_in))
        for j in range(len(args)):
            if j < self.n_g_out:
                name = self.g_out[j]
                try:
                    kwargs[name] = args[j]
                except:
                    raise Exception("Generic parameter " + str(name) + " set twice.")
            elif j < self.n_g_out + self.n_in:
                if type(args[j]) == WorkspaceVariable:
                    ins[j - self.n_g_out] = args[j].ws_id
                else:
                    temps.append(ws.add_variable(args[j]))
                    ins[j - self.n_g_out] = temps[-1].ws_id
            elif j < self.n_g_out + self.n_in + self.n_g_in:
                name = self.g_in[j - self.n_g_out - self.n_in]
                try:
                    kwargs[name] = args[j]
                except:
                    raise Exception("Generic parameter " + str(name) + " set twice.")
            else:
                raise Exception(str(j) + " positional arguments given, but this WSM expects " +
                                str(j-1) + ".")

        # Check output argument names
        g_output_args = dict()
        for k in self.g_out:
            if not k in kwargs:
                raise Exception("WSM " + self.name + " needs generic output " + k)
            else:
                g_output_args[k] = kwargs[k]

        # Check input argument names
        g_input_args = dict()
        for k in self.g_in:
            if not k in kwargs:
                if k in self.g_in_default:
                    g_input_args[k] = self.g_in_default[k]
                else:
                    raise Exception("WSM " + self.name + " needs generic input " + k)
            else:
                g_input_args[k] = kwargs[k]

        # Resolve overload (if necessary).
        g_out_types = dict([(k,WorkspaceVariable.get_group_id(g_output_args[k]))
                            for k in self.g_out])
        g_in_types  = dict([(k,WorkspaceVariable.get_group_id(g_input_args[k]))
                            for k in self.g_in])
        m_id = self.m_ids[0]
        sg_index = 0
        if (len(self.m_ids) > 1):
            if not g_in_types in self.g_in_types or not g_out_types in self.g_out_types:
                raise ValueError("Could not resolve call to supergeneric function.")
            else:
                out_index = self.g_out_types.index(g_out_types)
                in_index = self.g_in_types.index(g_in_types)
                m_id_out = self.m_ids[self.g_out_types.index(g_out_types)]
                m_id_in = self.m_ids[self.g_in_types.index(g_in_types)]
                if not out_index == in_index:
                    if self.g_in_types[in_index] == self.g_in_types[out_index]:
                        m_id = self.m_ids[out_index]
                        sg_index = out_index
                    elif self.g_out_types[out_index] == self.g_out_types[in_index]:
                        m_id = self.m_ids[in_index]
                        sg_index = in_index
                    else:
                        raise Exception("Could not uniquely resolve super-generic overload.")
                else:
                    m_id     = m_id_out
                    sg_index = out_index

        # Combine input and output arguments into lists.
        arts_args_out = []
        for out in self.outs:
            arts_args_out.append(out)

        for name in self.g_out:
            arg = g_output_args[name]
            if not type(arg) == WorkspaceVariable:
                raise ValueError("Generic Output " + name + " must be an ARTS WSV.")
            group_id = arg.group_id
            expected = self.g_out_types[sg_index][name]
            if not group_id == expected:
                raise Exception("Generic output " + name + " expected to be of type "
                                + group_names[expected])
            arts_args_out.append(arg.ws_id)

        arts_args_in = []
        for i in ins:
            if not i in self.outs:
                arts_args_in.append(i)

        for name in self.g_in:
            arg = g_input_args[name]
            if type(arg) == WorkspaceVariable:
                arts_args_in.append(arg.ws_id)
            else:
                group_id = WorkspaceVariable.get_group_id(arg)
                expected = self.g_in_types[sg_index][name]
                if not group_id == expected:
                    raise Exception("Generic input " + name + " expected to be of type "
                                    + group_names[expected])
                temps.append(ws.add_variable(arg))
                arts_args_in.append(temps[-1].ws_id)
        return (m_id, arts_args_out, arts_args_in, temps)


    def create(self, ws, name = None):
        """
        Call to <Group>Create WSMs are handled differently. This method simply
        determines the group type from the function name and then add a variable of
        this type to the workspace ws. A handle of this variable is then added to
        as attribute to the typhon.arts.workspace.variables module.

        Args:
            ws(Workspace): Workspace object to add the variable to
            name(str):     Name of the variable to add to the workspace
        """
        if not name:
            name = "__anonymous_" + str(len(ws.vars))
        group = WorkspaceMethod.create_regexp.match(self.name).group(1)
        group_id = group_ids[group]
        ws_id = arts_api.add_variable(ws.ptr, group_id)
        wsv = WorkspaceVariable(ws_id, name, group, "User defined variable.", ws)
        setattr(variables, name, wsv)
        ws.vars[name] = wsv
        return wsv

    def call(*args, **kwargs):
        """ Execute workspace method.

        This method will execute the workspace method (args[0]) on the workspace object (args[1])
        interpreting the remaining arguments in `*args` and `**kwargs` as arguments.

        Positional arguments in `*args` are interpreted in order with output arguments coming
        first.

        Keyword arguments in kwargs are interpreted according to the name of the generic
        parameters of the ARTS WSM.

        Args:
        args(list): Positional arguments with the first argument being the WorkspaceMethod
        instance, i.e. self = args[0], the second the Workspace object (args[1]). The
        remaining arguments are interpreted as generic arguments to the ARTS WSM.
        kargs(dict): Keyword args are interpreted as named generic arguments to the ARTS WSM
        according to its definition in methods.cc.
        """

        self = args[0]

        if self.is_create:
            return self.create(*args[1:])

        ws   = args[1]

        (m_id, arts_args_out, arts_args_in, temps) = self._parse_output_input_lists(ws,
                                                                                    args[2:],
                                                                                    kwargs)

        # Execute WSM and check for errors.
        arg_out_ptr = c.cast((c.c_long * len(arts_args_out))(*arts_args_out), c.POINTER(c.c_long))
        arg_in_ptr = c.cast((c.c_long * len(arts_args_in))(*arts_args_in), c.POINTER(c.c_long))
        print("out:" + str(arts_args_out))
        print("in: " + str(arts_args_in))

        e_ptr = arts_api.execute_workspace_method(ws.ptr, m_id,
                                                  len(arts_args_out),
                                                  arg_out_ptr,
                                                  len(arts_args_in),
                                                  arg_in_ptr)
        if (e_ptr):
            raise Exception("Call to ARTS WSM " + self.name + " failed with error: "
                           + e_ptr.decode("utf8").format())

        # Remove temporaries from workspace (in reverse order).
        for t in temps[::-1]:
            t.erase()


    def describe(self):
        """ Print WSM documentation. """
        print(self.description.format())

def iter_raw():
    """ Iterator returning a WorkspaceMethod object for each available ARTS WSM.

    This iterator returns super-generically overloaded methods several times.

    Yields:
        WorkspaceMethod: The next ARTS Workspace method as defined in methods.cc in
                         increasing order.
    """
    for i in range(arts_api.get_number_of_methods()):
        m            = arts_api.get_method(i)
        name         = m.name.decode("utf8")
        description  = m.description.decode("utf8")
        outs         = [m.outs[i] for i in range(m.n_out)]
        g_out_types  = [m.g_out_types[i] for i in range(m.n_g_out)]
        ins          = [m.ins[i] for i in range(m.n_in)]
        g_in_types   = [m.g_in_types[i] for i in range(m.n_g_in)]
        yield WorkspaceMethod(m.id, name, description, outs, g_out_types, ins, g_in_types)

def iter():
    """ Iterator returning a WorkspaceMethod object for each available ARTS WSM.

    This iterator returns overloaded Workspace methods, i.e. super-generically overloaded
    WSM are not returned multiple times.

    Yields:
        WorkspaceMethod: The next ARTS Workspace method as defined in methods.cc in
                         increasing order.
    """
    for k,m in workspace_methods:
        yield m

workspace_methods = dict()
for m in iter_raw():
    if m.name in workspace_methods:
        workspace_methods[m.name].add_overload(m.m_ids, m.g_in_types, m.g_out_types)
    else:
        workspace_methods[m.name] = m
