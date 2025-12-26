import copy
import math
from typing import Any, Iterable, List, Sequence, Tuple


def _validate_depth(depth: int | None, allow_none: bool) -> None:
    if depth is None:
        if not allow_none:
            raise TypeError("depth must be an int")
        return
    if not isinstance(depth, int):
        raise TypeError("depth must be an int or None")
    if depth < 0:
        raise ValueError("depth must be >= 0")


def _validate_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    try:
        shape = tuple(int(d) for d in shape)
    except Exception as e:
        raise TypeError("shape must be a sequence of integers") from e
    if any(d < 0 for d in shape):
        raise ValueError("shape dimensions must be non-negative")
    return shape


def lget_shape(nested: Any, depth: int | None = None) -> Tuple[int, ...]:
    """
    Infer the (prefix) shape of a nested list up to a given depth, and
    verify grid structure within those inspected levels.

    Args:
        nested: A nested Python list structure (or a scalar leaf).
        depth: How many list-levels to check/collect.
            - None: check fully (all levels).
            - 0: stop immediately; return ().
            - k > 0: require exactly k list levels and uniform lengths.

    Returns:
        A tuple of integers representing the shape prefix.

    Raises:
        ValueError: if the structure is not a proper grid within the inspected
                    depth, or if a non-list is encountered while depth > 0.
        TypeError: if depth is not int or None.
    """
    _validate_depth(depth, allow_none=True)

    if depth == 0:
        return ()

    if not isinstance(nested, list):
        # If depth is None, we are at a scalar leaf: empty shape suffix.
        if depth is None:
            return ()
        # Otherwise we expected more list levels.
        raise ValueError("Expected a list, but found a scalar before depth ran out")

    if not nested:  # empty list at this level
        # Once a dimension is zero, deeper dims (if any) must also be zero.
        # Returning (0,) is sufficient and standard.
        return (0,)

    # Recurse into first element; reduce depth if provided.
    next_depth = None if depth is None else depth - 1
    child_shape = lget_shape(nested[0], next_depth)

    # Verify all siblings match the child shape within inspected depth.
    for elem in nested[1:]:
        if lget_shape(elem, next_depth) != child_shape:
            raise ValueError("Irregular structure: not a proper grid")

    return (len(nested),) + child_shape


def _validate_grid_structure(nested: Any, depth: int | None = None) -> None:
    lget_shape(nested, depth=depth)


def lflatten(
    nested: Any, depth: int | None = None, return_reduced_shape: bool = False
) -> Tuple[List[Any], Tuple[int, ...]]:
    """
    Flatten a nested list with grid structure up to a given depth.
    Merges the first depth dimensions.

    Args:
        nested: A nested Python list structure (or a scalar leaf).
        depth:
            - None: flatten completely to 1D.
            - 0: no flattening; returns [nested] if nested is scalar, or nested as-is.
            - 1: shape (d0, d1, ... ) -> (d0 * d1, ... )
            - k > 0: flatten exactly k list levels.

    Returns:
        flat: The flattened list (to the specified depth).
        shape: The (prefix) shape that was validated/collected.

    Raises:
        ValueError, TypeError: bubbled up from lget_shape or input validation.
    """
    # Validate & collect the (prefix) shape to the requested depth.
    shape = lget_shape(nested, depth=depth)

    # If full flatten requested, use full rank as the depth to flatten.
    if depth is None:
        depth = len(shape)

    _validate_depth(depth, allow_none=False)  # already int here

    flat: List[Any] = []

    def _collect(x: Any, d: int) -> None:
        if d == -1 or not isinstance(x, list):
            flat.append(x)
        else:
            for elem in x:
                _collect(elem, d - 1)

    _collect(nested, depth)
    if return_reduced_shape:
        return flat, shape
    else:
        return flat


def linit(shape: Iterable[int], value: Any = None) -> Any:
    """
    Create a nested list with the given shape, filled with the specified value.
    Works for arbitrary Python objects (strings, numbers, etc.).

    Args:
        shape: Target shape (sequence of non-negative ints). Use () for a scalar.
        value: Value to fill the structure with.

    Returns:
        A nested list (or scalar if shape == ()) with the specified shape.

    Raises:
        TypeError: if shape is not a sequence of integers.
        ValueError: if any dim is negative.
    """
    shape = _validate_shape(shape)

    # Special case: scalar leaf (shape == ())
    if len(shape) == 0:
        return value

    def build(shp: Tuple[int, ...]) -> Any:
        if len(shp) == 1:
            return [value for _ in range(shp[0])]
        out: List[Any] = []
        dim = shp[0]
        rest = shp[1:]
        for _ in range(dim):
            chunk = build(rest)
            out.append(chunk)
        return out

    return build(shape)


def lunflatten(lst: List[Any], shape: Iterable[int]) -> Any:
    """
    Unflatten a flat list into nested lists with the given shape.
    Works for arbitrary Python objects (strings, numbers, etc.).

    Args:
        lst: Flat list of elements.
        shape: Target shape (sequence of non-negative ints). Use () for a scalar.

    Returns:
        A nested list (or scalar if shape == ()) with the specified shape.

    Raises:
        TypeError: if shape is not a sequence of integers.
        ValueError: if any dim is negative, or len(lst) != product(shape).
    """
    shape = _validate_shape(shape)

    total = math.prod(shape) if len(shape) > 0 else 1

    if len(lst) != total:
        raise ValueError(
            f"List length ({len(lst)}) does not match product(shape) ({total})"
        )

    # Special case: scalar leaf (shape == ())
    if len(shape) == 0:
        # Expect exactly one element; return it as a scalar.
        return lst[0]

    # Build nested structure in one pass using an index cursor.
    def build(index: int, shp: Tuple[int, ...]) -> tuple[Any, int]:
        if len(shp) == 1:
            # Slice the final run (no extra checks needed; total length already validated)
            end = index + shp[0]
            return lst[index:end], end
        out: List[Any] = []
        dim = shp[0]
        rest = shp[1:]
        for _ in range(dim):
            chunk, index = build(index, rest)
            out.append(chunk)
        return out, index

    result, final_idx = build(0, shape)
    # final_idx must equal total if everything was consumed correctly
    assert final_idx == total, "Internal error: index mismatch during unflatten"
    return result


def lsqueeze(nested: Any, depth: int) -> Any:
    """
    Remove the singleton axis at the given `depth` (0 = outermost).
    Only checks/assumes grid structure up to `depth+1` levels.

    Args:
        nested: nested list grid
        depth: axis index to remove (0-based, no negatives)

    Returns:
        nested list with that axis removed

    Raises:
        ValueError: if there are fewer than `depth+1` list levels,
                    or the axis at `depth` is not singleton (size != 1),
                    or the structure is not a proper grid up to that depth.
    """
    _validate_depth(depth, allow_none=False)
    _validate_grid_structure(nested, depth=depth)

    # Verify prefix shape to exactly the axis we remove:
    # Need depth+1 list levels to exist and be regular.
    prefix_shape = lget_shape(nested, depth=depth + 1)
    if len(prefix_shape) <= depth:
        raise ValueError(
            f"Cannot squeeze at depth={depth}: structure has only {len(prefix_shape)} list levels"
        )
    if prefix_shape[depth] != 1:
        raise ValueError(
            f"Cannot squeeze axis at depth={depth}: size={prefix_shape[depth]} != 1"
        )

    def rec(x: Any, d: int) -> Any:
        if d == depth:
            # Axis to remove: x must be a list of length 1
            # Guaranteed by get_shape; still guard defensively.
            if not isinstance(x, list) or len(x) != 1:
                raise AssertionError("Internal inconsistency while squeezing")
            return x[0]
        # Above the target axis, we must be in list context (verified by get_shape)
        return [rec(e, d + 1) for e in x]  # type: ignore[index]

    return rec(nested, 0)


def lunsqueeze(nested: Any, depth: int) -> Any:
    """
    Insert a new axis of size 1 at position `depth` (0 = outermost).
    Only checks/assumes grid structure up to `depth` levels.

    Args:
        nested: nested list grid (or scalar if depth==0)
        depth: axis position to insert (0-based, no negatives)

    Returns:
        nested list with a new size-1 axis inserted at `depth`

    Raises:
        ValueError: if there are fewer than `depth` list levels to reach the
                    insertion point (i.e., cannot descend that far),
                    or the structure is not a proper grid up to that depth.
    """
    _validate_depth(depth, allow_none=False)
    _validate_grid_structure(nested, depth=depth)

    # Need to ensure we can descend `depth` list levels uniformly.
    prefix_shape = lget_shape(nested, depth=depth)
    if len(prefix_shape) < depth:
        raise ValueError(
            f"Cannot unsqueeze at depth={depth}: structure has only {len(prefix_shape)} list levels"
        )

    def rec(x: Any, d: int) -> Any:
        if d == depth:
            # Insert new axis here
            return [x]
        # Above insertion point, we must be in list context (verified by lget_shape)
        return [rec(e, d + 1) for e in x]  # type: ignore[index]

    return rec(nested, 0)


def _validate_times(times: int) -> None:
    if not isinstance(times, int):
        raise TypeError("times must be an int")
    if times < 0:
        raise ValueError("times must be >= 0")


def lrepeat(nested: Any, depth: int, times: int) -> Any:
    """
    Repeat (tile) the singleton axis at 0-based `depth` (0 = outermost)
    exactly `times` times. Deep-copies the repeated element to avoid aliasing.

    Only validates grid structure up to `depth+1` levels; deeper structure
    may be ragged and is left untouched.

    Args:
        nested: nested list grid
        depth: axis index to repeat (must exist and be size 1)
        times: number of repeats to create (>= 0)

    Returns:
        A new nested list with that axis size changed from 1 to `times`.

    Raises:
        ValueError: if the axis doesn’t exist, isn’t singleton, or structure
                    is not a proper grid up to `depth+1`.
        TypeError: for invalid argument types.
    """
    _validate_depth(depth, allow_none=False)
    _validate_grid_structure(nested, depth=depth)
    _validate_times(times)

    # Ensure axis exists and is singleton
    prefix_shape = lget_shape(nested, depth=depth + 1)
    if len(prefix_shape) <= depth:
        raise ValueError(
            f"Cannot repeat at depth={depth}: only {len(prefix_shape)} list levels"
        )
    if prefix_shape[depth] != 1:
        raise ValueError(
            f"Cannot repeat at depth={depth}: axis size is {prefix_shape[depth]} (not 1)"
        )

    def rec(x: Any, d: int) -> Any:
        if d == depth:
            # x must be a list with exactly one element (verified above)
            base = x[0]
            return [copy.deepcopy(base) for _ in range(times)]
        # Preserve outer axes; recurse deeper
        return [rec(e, d + 1) for e in x]  # type: ignore[index]

    return rec(nested, 0)


def largmax_filtered(lst, dummy_return=None):
    lst_filtered = [x for x in lst if x is not None]
    if len(lst_filtered) > 0:
        return lst.index(max(lst_filtered))
    else:
        return dummy_return


def largmin_filtered(lst, dummy_return=None):
    lst_filtered = [x for x in lst if x is not None]
    if len(lst_filtered) > 0:
        return lst.index(min(lst_filtered))
    else:
        return dummy_return


def lslice(lst: List[Any], start: int, end: int, axis: int) -> List[Any]:
    """
    Slice the nested list at the given depth (0 = outermost).
    Only checks/assumes grid structure up to `depth` levels.

    Args:
        lst: nested list grid
        start: slice start index (inclusive)
        end: slice end index (exclusive)
        depth: axis index to slice (0-based, no negatives)
    Returns:
        nested list with that axis sliced
    Raises:
        ValueError: if there are fewer than `depth+1` list levels,
                    or the structure is not a proper grid up to that depth.
    """

    depth = axis
    _validate_depth(depth, allow_none=False)
    _validate_grid_structure(lst, depth=depth)

    # Verify prefix shape to exactly the axis we slice:
    # Need depth+1 list levels to exist and be regular.
    prefix_shape = lget_shape(lst, depth=depth + 1)
    if len(prefix_shape) <= depth:
        raise ValueError(
            f"Cannot slice at depth={depth}: structure has only {len(prefix_shape)} list levels"
        )

    def rec(x: Any, d: int) -> Any:
        if d == depth:
            # Axis to slice: x must be a list
            # Guaranteed by get_shape; still guard defensively.
            if not isinstance(x, list):
                raise AssertionError("Internal inconsistency while slicing")
            # check that start/end are in range
            if start < 0 or end > len(x) or start > end:
                raise ValueError(
                    f"Cannot slice axis at depth={depth}: invalid slice [{start}:{end}] for size {len(x)}"
                )
            return x[start:end]
        # Above the target axis, we must be in list context (verified by get_shape)
        return [rec(e, d + 1) for e in x]  # type: ignore[index]

    return rec(lst, 0)
