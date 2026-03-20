"""
Loop infrastructure - required for Tile Loop. Registered but not in display mappings.
"""

try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
    from comfy_execution.graph import ExecutionBlocker
    _HAS_EXECUTION = True
except ImportError:
    _HAS_EXECUTION = False
    GraphBuilder = None
    is_link = lambda x: isinstance(x, (list, tuple)) and len(x) == 2
    ExecutionBlocker = None

NUM_FLOW_SOCKETS = 5


class AccumulateTile:
    """Appends to_add to accumulation; outputs the list (combines accumulate + to_list)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"to_add": ("*",)}, "optional": {"accumulation": ("*",)}}
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "accumulate"
    CATEGORY = "Video Tiler"

    def accumulate(self, to_add, accumulation=None):
        if accumulation is None:
            return ([to_add],)
        if isinstance(accumulation, dict) and "accum" in accumulation:
            lst = accumulation["accum"] + [to_add]
        elif isinstance(accumulation, (list, tuple)):
            lst = list(accumulation) + [to_add]
        else:
            lst = [to_add]
        return (lst,)


class IntMathOperation:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("INT", {"default": 0}), "b": ("INT", {"default": 0}),
            "operation": (["add", "subtract", "multiply", "divide", "modulo"],)}}
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("a",)
    FUNCTION = "op"
    CATEGORY = "Video Tiler"

    def op(self, a, b, operation):
        if operation == "add": return (a + b,)
        elif operation == "subtract": return (a - b,)
        elif operation == "multiply": return (a * b,)
        elif operation == "divide": return (a // b if b else 0,)
        elif operation == "modulo": return (a % b if b else 0,)
        return (a,)


class IntConditions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("INT", {"default": 0}), "b": ("INT", {"default": 0}),
            "operation": (["==", "!=", "<", ">", "<=", ">="],)}}
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "cond"
    CATEGORY = "Video Tiler"

    def cond(self, a, b, operation):
        if operation == ">" and b == 0:
            print(f"[Video Tiler] IntConditions: a={a!r} (type={type(a).__name__})")
        ops = {"==": lambda a,b: a==b, "!=": lambda a,b: a!=b, "<": lambda a,b: a<b,
               ">": lambda a,b: a>b, "<=": lambda a,b: a<=b, ">=": lambda a,b: a>=b}
        return (ops.get(operation, lambda a,b: False)(a, b),)


if _HAS_EXECUTION:

    class WhileLoopOpen:
        @classmethod
        def INPUT_TYPES(cls):
            inputs = {"required": {"condition": ("BOOLEAN", {"default": True})}, "optional": {}}
            for i in range(NUM_FLOW_SOCKETS):
                inputs["optional"][f"initial_value{i}"] = ("*",)
            return inputs
        RETURN_TYPES = tuple(["FLOW_CONTROL"] + ["*"] * NUM_FLOW_SOCKETS)
        RETURN_NAMES = tuple(["FLOW_CONTROL"] + [f"value{i}" for i in range(NUM_FLOW_SOCKETS)])
        FUNCTION = "open"
        CATEGORY = "Video Tiler"

        def open(self, condition, **kwargs):
            return tuple(["stub"] + [kwargs.get(f"initial_value{i}", None) for i in range(NUM_FLOW_SOCKETS)])

    class WhileLoopClose:
        @classmethod
        def INPUT_TYPES(cls):
            inputs = {"required": {"flow_control": ("FLOW_CONTROL", {"rawLink": True}), "condition": ("BOOLEAN", {"forceInput": True})},
                "optional": {}, "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}
            for i in range(NUM_FLOW_SOCKETS):
                inputs["optional"][f"initial_value{i}"] = ("*",)
            return inputs
        RETURN_TYPES = tuple(["*"] * NUM_FLOW_SOCKETS)
        RETURN_NAMES = tuple([f"value{i}" for i in range(NUM_FLOW_SOCKETS)])
        FUNCTION = "close"
        CATEGORY = "Video Tiler"

        def _explore_dependencies(self, node_id, dynprompt, upstream):
            node_info = dynprompt.get_node(node_id)
            if "inputs" not in node_info:
                return
            for k, v in node_info["inputs"].items():
                if is_link(v):
                    parent_id = v[0]
                    if parent_id not in upstream:
                        upstream[parent_id] = []
                        self._explore_dependencies(parent_id, dynprompt, upstream)
                    upstream[parent_id].append(node_id)

        def _collect_contained(self, node_id, upstream, contained):
            if node_id not in upstream:
                return
            for child_id in upstream[node_id]:
                if child_id not in contained:
                    contained[child_id] = True
                    self._collect_contained(child_id, upstream, contained)

        def close(self, flow_control, condition, dynprompt=None, unique_id=None, **kwargs):
            if not condition:
                print(f"[Video Tiler] WhileLoopClose: condition=False, exiting loop")
                return tuple(kwargs.get(f"initial_value{i}", None) for i in range(NUM_FLOW_SOCKETS))
            upstream = {}
            self._explore_dependencies(unique_id, dynprompt, upstream)
            contained = {}
            open_node = flow_control[0]
            self._collect_contained(open_node, upstream, contained)
            contained[unique_id] = True
            contained[open_node] = True
            graph = GraphBuilder()
            for node_id in contained:
                original = dynprompt.get_node(node_id)
                name = "Recurse" if node_id == unique_id else node_id
                node = graph.node(original["class_type"], name)
                node.set_override_display_id(node_id)
            for node_id in contained:
                original = dynprompt.get_node(node_id)
                node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
                for k, v in original["inputs"].items():
                    if is_link(v) and v[0] in contained:
                        parent = graph.lookup_node(v[0])
                        node.set_input(k, parent.out(v[1]))
                    else:
                        node.set_input(k, v)
            new_open = graph.lookup_node(open_node)
            for i in range(NUM_FLOW_SOCKETS):
                new_open.set_input(f"initial_value{i}", kwargs.get(f"initial_value{i}", None))
            my_clone = graph.lookup_node("Recurse")
            return {"result": tuple(my_clone.out(i) for i in range(NUM_FLOW_SOCKETS)), "expand": graph.finalize()}

    class ForLoopOpen:
        @classmethod
        def INPUT_TYPES(cls):
            inputs = {"required": {"remaining": ("INT", {"default": 1, "min": 0, "max": 100000})},
                "optional": {f"initial_value{i}": ("*",) for i in range(1, NUM_FLOW_SOCKETS)}, "hidden": {"initial_value0": ("*",)}}
            return inputs
        RETURN_TYPES = tuple(["FLOW_CONTROL", "INT"] + ["*"] * (NUM_FLOW_SOCKETS - 1))
        RETURN_NAMES = tuple(["flow_control", "remaining"] + [f"value{i}" for i in range(1, NUM_FLOW_SOCKETS)])
        FUNCTION = "open"
        CATEGORY = "Video Tiler"

        def open(self, remaining, **kwargs):
            graph = GraphBuilder()
            if "initial_value0" in kwargs:
                remaining = kwargs["initial_value0"]
            init_vals = {f"initial_value{i}": kwargs.get(f"initial_value{i}", None) for i in range(NUM_FLOW_SOCKETS)}
            init_vals["initial_value0"] = remaining
            while_open = graph.node("WhileLoopOpen", condition=remaining, **init_vals)
            return {"result": tuple(["stub", remaining] + [kwargs.get(f"initial_value{i}", None) for i in range(1, NUM_FLOW_SOCKETS)]), "expand": graph.finalize()}

    class ForLoopClose:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"flow_control": ("FLOW_CONTROL", {"rawLink": True})},
                "optional": {f"initial_value{i}": ("*", {"rawLink": True}) for i in range(1, NUM_FLOW_SOCKETS)}}
        RETURN_TYPES = tuple(["*"] * (NUM_FLOW_SOCKETS - 1))
        RETURN_NAMES = tuple([f"value{i}" for i in range(1, NUM_FLOW_SOCKETS)])
        FUNCTION = "close"
        CATEGORY = "Video Tiler"

        def close(self, flow_control, **kwargs):
            graph = GraphBuilder()
            while_open = flow_control[0]
            sub = graph.node("IntMathOperation", operation="subtract", a=[while_open, 1], b=1)
            cond = graph.node("IntConditions", a=[while_open, 1], b=0, operation=">")
            input_vals = {f"initial_value{i}": kwargs.get(f"initial_value{i}", None) for i in range(1, NUM_FLOW_SOCKETS)}
            while_close = graph.node("WhileLoopClose", flow_control=flow_control, condition=cond.out(0), initial_value0=sub.out(0), **input_vals)
            return {"result": tuple(while_close.out(i) for i in range(1, NUM_FLOW_SOCKETS)), "expand": graph.finalize()}

else:
    WhileLoopOpen = None
    WhileLoopClose = None
    ForLoopOpen = None
    ForLoopClose = None
