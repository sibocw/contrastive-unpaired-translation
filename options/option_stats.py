import inspect

class OptionsWrapper:
    def __init__(self, opt):
        object.__setattr__(self, 'opt', opt)
        object.__setattr__(self, 'opt_visited_count', {})
        object.__setattr__(self, 'queried_attribute_values', {})
        object.__setattr__(self, 'queried_attribute_lines', {})

    def __getattr__(self, name):
        if name not in self.opt_visited_count:
            self.opt_visited_count[name] = 0
            self.queried_attribute_values[name] = set()
            self.queried_attribute_lines[name] = []
        self.opt_visited_count[name] += 1
        val = getattr(self.opt, name)
        hashable_val = val if isinstance(val, (int, float, str)) else str(val)
        self.queried_attribute_values[name].add(hashable_val)
        # Record the line number and filename of the caller
        frame = inspect.currentframe()
        outer_frames = inspect.getouterframes(frame)
        # The caller is outer_frames[1]
        caller_info = outer_frames[1]
        filename = caller_info.filename
        lineno = caller_info.lineno
        self.queried_attribute_lines[name].append((filename, lineno))
        return val

    def __setattr__(self, name, value):
        if name in ['opt', 'opt_visited_count']:
            # Set wrapper's own attributes directly
            object.__setattr__(self, name, value)
        else:
            # Set attributes on the wrapped object
            setattr(self.opt, name, value)
    
    @property
    def __dict__(self):
        # Return the __dict__ of the wrapped object for functions like vars()
        return self.opt.__dict__
    
    def __len__(self):
        # Return the number of attributes in the wrapped object
        return len(self.opt_visited_count)
    
    def print_summary(self):
        print(f"{len(self)} option attributes visited:")
        for name, count in self.opt_visited_count.items():
            print(name)
            print(f"  count: {count}")
            print(f"  values: {self.queried_attribute_values[name]}")
            print(f"  lines: {self.queried_attribute_lines[name]}")
            print()