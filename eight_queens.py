"""
Exploring the Eight Queens Problem
==================================

Place eight queens on an 8x8 chessboard so that no two queens attack each
other: no two share a row, a column, or a diagonal.

This file explores the problem in a few ways:
  1. Brute force over permutations (simple, but slower)
  2. Backtracking (the classic approach)
  3. Bitmask backtracking (fast, scales to larger boards)
  4. Symmetry: reducing the 92 solutions to 12 "fundamental" ones
  5. Min-conflicts local search (finds one solution fast, even for big n)

Run it with:  python eight_queens.py
"""

import random
import time
from itertools import permutations

# A solution is represented as a tuple `cols` of length n, where cols[r] is
# the column of the queen in row r. Using one queen per row (and a
# permutation of columns) automatically rules out row and column conflicts,
# so we only ever need to check diagonals.


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def board_string(cols):
    """Return a text picture of the board for a given solution."""
    n = len(cols)
    lines = []
    for r in range(n):
        row = ["Q" if cols[r] == c else "." for c in range(n)]
        lines.append(" ".join(row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Brute force over permutations
# ---------------------------------------------------------------------------

def is_valid(cols):
    """Check that no two queens share a diagonal."""
    n = len(cols)
    for r1 in range(n):
        for r2 in range(r1 + 1, n):
            if abs(cols[r1] - cols[r2]) == r2 - r1:
                return False
    return True


def solve_brute_force(n=8):
    """Try all n! column permutations and keep the valid ones."""
    return [p for p in permutations(range(n)) if is_valid(p)]


# ---------------------------------------------------------------------------
# 2. Backtracking
# ---------------------------------------------------------------------------

def solve_backtracking(n=8):
    """Place queens row by row, abandoning a branch as soon as it conflicts."""
    solutions = []
    cols = []
    used_cols = set()
    used_diag = set()   # r - c is constant along "\" diagonals
    used_anti = set()   # r + c is constant along "/" diagonals

    def place(r):
        if r == n:
            solutions.append(tuple(cols))
            return
        for c in range(n):
            if c in used_cols or (r - c) in used_diag or (r + c) in used_anti:
                continue
            cols.append(c)
            used_cols.add(c)
            used_diag.add(r - c)
            used_anti.add(r + c)

            place(r + 1)

            cols.pop()
            used_cols.remove(c)
            used_diag.remove(r - c)
            used_anti.remove(r + c)

    place(0)
    return solutions


# ---------------------------------------------------------------------------
# 3. Bitmask backtracking (counting only)
# ---------------------------------------------------------------------------

def count_bitmask(n=8):
    """Count solutions using bit operations to track attacked squares.

    `cols`, `diag`, and `anti` are bitmasks of columns under attack in the
    current row. Diagonal attacks shift by one column each row down.
    """
    full = (1 << n) - 1

    def search(cols, diag, anti):
        if cols == full:
            return 1
        total = 0
        free = full & ~(cols | diag | anti)
        while free:
            bit = free & -free          # lowest available column
            free ^= bit
            total += search(cols | bit,
                            ((diag | bit) << 1) & full,
                            (anti | bit) >> 1)
        return total

    return search(0, 0, 0)


# ---------------------------------------------------------------------------
# 4. Symmetry: fundamental solutions
# ---------------------------------------------------------------------------

def rotate(cols):
    """Rotate the board 90 degrees clockwise."""
    n = len(cols)
    new = [0] * n
    for r, c in enumerate(cols):
        new[c] = n - 1 - r
    return tuple(new)


def reflect(cols):
    """Mirror the board left-to-right."""
    n = len(cols)
    return tuple(n - 1 - c for c in cols)


def symmetries(cols):
    """All 8 images of a solution under rotations and reflections."""
    images = []
    current = tuple(cols)
    for _ in range(4):
        images.append(current)
        images.append(reflect(current))
        current = rotate(current)
    return images


def fundamental_solutions(solutions):
    """Group solutions into symmetry classes; keep one representative each."""
    seen = set()
    fundamentals = []
    for sol in solutions:
        if sol in seen:
            continue
        images = symmetries(sol)
        seen.update(images)
        fundamentals.append(min(images))
    return sorted(fundamentals)


# ---------------------------------------------------------------------------
# 5. Min-conflicts local search
# ---------------------------------------------------------------------------

def conflicts(cols, row, col):
    """Number of queens attacking square (row, col), ignoring row's own queen."""
    return sum(
        1 for r, c in enumerate(cols)
        if r != row and (c == col or abs(c - col) == abs(r - row))
    )


def solve_min_conflicts(n=8, max_steps=10_000, seed=None):
    """Start from a random board and repeatedly move a conflicted queen to
    the column in its row with the fewest conflicts.

    Unlike backtracking, this finds a single solution very quickly even for
    boards with hundreds of queens.
    """
    rng = random.Random(seed)
    cols = [rng.randrange(n) for _ in range(n)]
    for step in range(max_steps):
        conflicted = [r for r in range(n) if conflicts(cols, r, cols[r]) > 0]
        if not conflicted:
            return tuple(cols), step
        row = rng.choice(conflicted)
        scores = [conflicts(cols, row, c) for c in range(n)]
        best = min(scores)
        cols[row] = rng.choice([c for c in range(n) if scores[c] == best])
    return None, max_steps


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def main():
    print("=" * 50)
    print("THE EIGHT QUEENS PROBLEM")
    print("=" * 50)

    brute, t_brute = timed(solve_brute_force, 8)
    back, t_back = timed(solve_backtracking, 8)
    bits, t_bits = timed(count_bitmask, 8)

    print("\nComparing methods on an 8x8 board:")
    print(f"  Brute force (8! = 40,320 boards): {len(brute):>3} solutions in {t_brute:.4f}s")
    print(f"  Backtracking:                     {len(back):>3} solutions in {t_back:.4f}s")
    print(f"  Bitmask backtracking:             {bits:>3} solutions in {t_bits:.4f}s")
    assert sorted(brute) == sorted(back) and bits == len(back)

    print("\nThe first solution found by backtracking:")
    print(board_string(back[0]))

    fundamentals = fundamental_solutions(back)
    print(f"\nUp to rotation and reflection, the {len(back)} solutions "
          f"reduce to {len(fundamentals)} fundamental ones:")
    for i, sol in enumerate(fundamentals, 1):
        n_images = len(set(symmetries(sol)))
        print(f"  {i:>2}. {sol}  ({n_images} distinct images)")
    print("  (One fundamental solution is symmetric under 180-degree rotation,"
          " so it has only 4 images: 11*8 + 1*4 = 92.)")

    print("\nHow the count grows with board size (n-queens):")
    print("   n | solutions | fundamental")
    print("  ---+-----------+------------")
    for n in range(1, 11):
        sols = solve_backtracking(n)
        print(f"  {n:>2} | {len(sols):>9} | {len(fundamental_solutions(sols)):>11}")
    print("  Note: n = 2 and n = 3 have no solutions at all.")

    print("\nMin-conflicts local search on bigger boards:")
    for n in (8, 50, 100):
        (sol, steps), t = timed(solve_min_conflicts, n, seed=1)
        status = "solved" if sol and is_valid(sol) else "gave up"
        print(f"  n = {n:>3}: {status} in {steps:>4} steps ({t:.3f}s)")


if __name__ == "__main__":
    main()
