import random
from collections import defaultdict
import json
import os
import argparse
import copy


class Solution:
    # Our protocal: 1~(N-1)^2 represnets rings, the last bit represents direction.
    def __init__(self, N=16, k=8, p=0.01, lam=0.01, lam_1=1, objective='default'):
        self.lam = lam
        self.lam_1 = lam_1
        self.objective = objective
        self.N = N
        self.k = k
        self.p = p
        self.binary_strings = []
        self.ring_vertices = {}
        self.ring_edges = {}
        self.edge_count = {}
        for index in range(self.k):
            self.ring_vertices[index] = []
            self.ring_edges[index] = []
        self.num_digits = (N - 1) ** 2 + 1
        for _ in range(k):
            binary_string = ''.join(random.choice('01')
                                    for _ in range(self.num_digits))
            self.binary_strings.append(binary_string)
        for _ in range(k):
            self.retain_connected_component(_)
            self.compute_rings()

    def compute_rings(self):
        for index in range(self.k):
            binary_string = self.binary_strings[index]
            grid_size = self.N - 1
            edges = set()
            vertices = set()

            for i in range(grid_size):
                for j in range(grid_size):
                    # Current grid point
                    current_index = i * grid_size + j
                    if binary_string[current_index] == '1':
                        # Check right edge
                        if j + 1 < grid_size:
                            right_index = i * grid_size + (j + 1)
                            if binary_string[right_index] == '0':
                                edges.add(((j + 1, i), (j + 1, i + 1)))
                                vertices.update([(j + 1, i), (j + 1, i + 1)])
                        else:
                            edges.add(((j + 1, i), (j + 1, i + 1)))
                            vertices.update([(j + 1, i), (j + 1, i + 1)])

                        # Check bottom edge
                        if i + 1 < grid_size:
                            bottom_index = (i + 1) * grid_size + j
                            if binary_string[bottom_index] == '0':
                                edges.add(((j, i + 1), (j + 1, i + 1)))
                                vertices.update([(j, i + 1), (j + 1, i + 1)])
                        else:
                            edges.add(((j, i + 1), (j + 1, i + 1)))
                            vertices.update([(j, i + 1), (j + 1, i + 1)])

                        # Check left edge
                        if j - 1 >= 0:
                            left_index = i * grid_size + (j - 1)
                            if binary_string[left_index] == '0':
                                edges.add(((j, i), (j, i + 1)))
                                vertices.update([(j, i), (j, i + 1)])
                        else:
                            edges.add(((j, i), (j, i + 1)))
                            vertices.update([(j, i), (j, i + 1)])

                        # Check top edge
                        if i - 1 >= 0:
                            top_index = (i - 1) * grid_size + j
                            if binary_string[top_index] == '0':
                                edges.add(((j, i), (j + 1, i)))
                                vertices.update([(j, i), (j + 1, i)])
                        else:
                            # Add top edge if on boundary
                            edges.add(((j, i), (j + 1, i)))
                            vertices.update([(j, i), (j + 1, i)])

            edge_loop = []
            if edges:
                start_edge = edges.pop()
                edge_loop.append(start_edge)

                current_edge = start_edge
                while edges:
                    next_edge = None
                    for edge in edges:
                        if current_edge[1] in edge:
                            next_edge = edge
                            break

                    if next_edge:
                        edges.remove(next_edge)
                        if next_edge[0] == current_edge[1]:
                            edge_loop.append(next_edge)
                        else:
                            edge_loop.append((next_edge[1], next_edge[0]))
                        current_edge = edge_loop[-1]
                    else:
                        break

            if edge_loop == []:
                continue
            vertex_loop = [edge_loop[0][0]]
            for edge in edge_loop:
                vertex_loop.append(edge[1])

            if vertex_loop[0] != vertex_loop[-1]:
                vertex_loop.append(vertex_loop[0])

            # Identify the leftmost vertical edge
            leftmost_vertical_edge = min(
                [edge for edge in edge_loop if edge[0][0]
                    == edge[1][0]],  # Vertical edges only
                key=lambda e: e[0][0]  # Sort by x-coordinate
            )

            if binary_string[-1] == '1':  # cw
                if leftmost_vertical_edge[0][1] < leftmost_vertical_edge[1][1]:
                    vertex_loop.reverse()
            if binary_string[-1] == '0':  # ccw
                if leftmost_vertical_edge[0][1] > leftmost_vertical_edge[1][1]:
                    vertex_loop.reverse()

            self.ring_vertices[index] = vertex_loop
            self.ring_edges[index] = edge_loop
            self.count_edges_in_rings()

    def retain_connected_component(self, index):
        if 0 <= index <= self.k-1:
            binary_string = self.binary_strings[index]
            grid_size = self.N - 1

            if '1' not in binary_string[0:-1]:
                self.ring_vertices[index] = []
                self.ring_edges[index] = []
                self.edge_count = {}
                return

            visited = [False] * (self.num_digits - 1)

            # Find a starting point
            max_attempts = 1000
            attempts = 0
            while attempts < max_attempts:
                start = random.randint(0, grid_size**2 - 1)
                if binary_string[start] == '1':
                    break
                attempts += 1
            if attempts >= max_attempts:
                self.ring_vertices[index] = []
                self.ring_edges[index] = []
                self.edge_count = {}
                print(
                    f"Warning: No valid starting point found for index {index} in string {binary_string}. Exiting early.")
                return

            # BFS
            queue = [start]
            visited[start] = True
            connected_component = ['0'] * (self.num_digits - 1)
            connected_component[start] = '1'

            while queue:
                current = queue.pop(0)
                neighbors = self.get_neighbors(current)
                for neighbor in neighbors:
                    if binary_string[neighbor] == '1' and not visited[neighbor]:
                        visited[neighbor] = True
                        connected_component[neighbor] = '1'
                        queue.append(neighbor)

            # New binary string
            new_binary_string = ''.join(
                connected_component) + binary_string[-1]
            self.binary_strings[index] = new_binary_string

        else:
            raise IndexError("Index out of range. Must be between 1 and k.")

    def get_neighbors(self, index):
        grid_size = self.N - 1
        neighbors = []
        row, col = divmod(index, grid_size)
        if row > 0:
            neighbors.append(index - grid_size)
        if row < grid_size - 1:
            neighbors.append(index + grid_size)
        if col > 0:
            neighbors.append(index - 1)
        if col < grid_size - 1:
            neighbors.append(index + 1)
        return neighbors

    def visualize(self, index):
        if 0 <= index <= self.k-1:
            binary_string = self.binary_strings[index]
            grid_size = self.N - 1
            grid = [[int(binary_string[i * grid_size + j])
                     for j in range(grid_size)] for i in range(grid_size)]
            for row in grid:
                print(row)
        else:
            raise IndexError("Index out of range. Must be between 1 and k.")

    def visualize_ring(self, index, output_file=None):
        if index not in self.ring_vertices:
            print(f"Index {index} not found in ring_vertices.")
            return

        grid_size = self.N
        grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]

        vertices = self.ring_vertices[index]

        # 如果 vertices 列表为空，则直接输出网格，不做箭头标记
        if not vertices:
            if output_file:
                with open(output_file, 'a') as f:
                    for row in grid:
                        f.write(' '.join(row) + '\n')
                    f.write("\n")
            else:
                for row in grid:
                    print(' '.join(row))
            return

        # 绘制环的路径
        for i in range(len(vertices) - 1):
            x1, y1 = vertices[i]
            x2, y2 = vertices[i + 1]

            if x1 == x2 and y1 < y2:
                grid[y1][x1] = '↓'
            elif x1 == x2 and y1 > y2:
                grid[y1][x1] = '↑'
            elif y1 == y2 and x1 < x2:
                grid[y1][x1] = '→'
            elif y1 == y2 and x1 > x2:
                grid[y1][x1] = '←'

        # 标记最后一段回到起点的方向
        x1, y1 = vertices[-1]
        x2, y2 = vertices[0]
        if x1 == x2 and y1 < y2:
            grid[y1][x1] = '↓'
        elif x1 == x2 and y1 > y2:
            grid[y1][x1] = '↑'
        elif y1 == y2 and x1 < x2:
            grid[y1][x1] = '→'
        elif y1 == y2 and x1 > x2:
            grid[y1][x1] = '←'

        if output_file:
            # 输出到文件
            with open(output_file, 'a') as f:
                for row in grid:
                    f.write(' '.join(row) + '\n')
                f.write("\n")
            print(f"Visualized ring saved to {output_file}")
        else:
            # 打印到命令行
            for row in grid:
                print(' '.join(row))

    def mutation(self, idx):
        if 0 <= idx <= self.k-1:
            binary_string = list(self.binary_strings[idx])
            for i in range(self.num_digits - 1):
                if random.random() < self.p:
                    binary_string[i] = '0' if binary_string[i] == '1' else '1'
            self.binary_strings[idx] = ''.join(binary_string)
            self.retain_connected_component(idx)
            self.compute_rings()

    def compute_distance(self):
        distance = 0
        for xa in range(self.N):
            for ya in range(self.N):
                for xb in range(self.N):
                    for yb in range(self.N):
                        # 对于每一对A,B点来说：
                        # 如果两个点重合了，就跳过
                        if xa == xb and ya == yb:
                            continue
                        distance_ab = []
                        # 计算每个环上A,B的距离
                        for index in range(self.k):
                            ring = self.ring_vertices[index]
                            distance_ab.append(
                                self.compute_distance_of_pair(xa, ya, xb, yb, ring))
                        # 选出最小的距离，作为distance_ab的最终距离
                        distance += min(distance_ab)

        distance /= (self.N ** 2) * (self.N ** 2 - 1)
        return distance

    def compute_distance_of_pair(self, xa, ya, xb, yb, ring):
        point_a = (xa, ya)
        point_b = (xb, yb)

        if point_a not in ring or point_b not in ring:
            return 100000

        index_a = ring.index(point_a)
        index_b = ring.index(point_b)

        if index_a <= index_b:
            distance_ab = index_b - index_a
        else:
            distance_ab = len(ring) - (index_a - index_b) - 1

        return distance_ab

    def evaluate(self):
        objective = self.objective
        lenth = 0
        for index in range(self.k):
            lenth += len(self.ring_vertices[index]) - 1  # Lenth of each ring

        distance = self.compute_distance()

        score = distance + self.lam * lenth

        variance = 0
        if objective == 'variance':
            variance = self.compute_variance()
            score += self.lam_1 * (variance ** 0.5)

        return score, distance, lenth, variance

    def compute_variance(self):
        overlapping = list(self.edge_count.values())
        if len(overlapping) == 0:
            return 0

        average = sum(overlapping) / len(overlapping)

        variance = sum((x - average) ** 2 for x in overlapping) / \
            len(overlapping)

        return variance

    def get_binary_string(self, index):
        if 0 <= index <= self.k-1:
            return self.binary_strings[index]
        else:
            raise IndexError("Index out of range. Must be between 1 and k.")

    def count_edges_in_rings(self):
        rings = list(self.ring_vertices.values())
        edge_count = {}

        for ring in rings:
            for i in range(len(ring)-1):
                start = ring[i]
                end = ring[i+1]

                edge = (start, end)  # Sequential

                if edge in edge_count:
                    edge_count[edge] += 1
                else:
                    edge_count[edge] = 1

        self.edge_count = edge_count
        return edge_count

    def return_ring(self):
        output = []
        for i in range(self.k):
            output.append(self.ring_vertices[i])
        return output


def evolve_population(N, k, p, lam, lam_1, objective, population, file_path, num_solution, gen_num=100):
    for t in range(gen_num):
        tem_population = []

        for solution in population:
            new_solution = copy.deepcopy(solution)
            for _ in range(k):
                new_solution.mutation(_)
            tem_population.append(new_solution)

        combined_population = population + tem_population

        combined_population.sort(key=lambda sol: sol.evaluate()[0])
        population = combined_population[:num_solution]

        best_score, distance, lenth, variance = combined_population[0].evaluate(
        )
        print(
            f"Generation {t}, best score: {best_score}, among which distance: {distance}, lenth: {lenth}, variance: {variance}")

        if (t + 1) % 100 == 0:
            save_to_file(combined_population[0], t + 1, file_path=file_path)

    return population


def save_to_file(solution, generation, file_path='population_data.txt'):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_path, 'a') as outfile:
        outfile.write(f"Generation {generation}\n")
        outfile.write("Binary Strings:\n")
        for idx, binary_string in enumerate(solution.binary_strings):
            outfile.write(f"{idx}: {binary_string}\n")

        outfile.write("Ring Vertices:\n")
        for key, vertices in solution.ring_vertices.items():
            vertex_str = ' -> '.join([f"({x},{y})" for x, y in vertices])
            outfile.write(f"{key}: {vertex_str}\n")

        outfile.write("Ring Visualizations:\n")
        for key in solution.ring_vertices.keys():
            outfile.write(f"Visualization of ring {key}:\n")

            grid_size = solution.N
            grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]

            vertices = solution.ring_vertices[key]
            if vertices:
                for i in range(len(vertices) - 1):
                    x1, y1 = vertices[i]
                    x2, y2 = vertices[i + 1]

                    if x1 == x2 and y1 < y2:
                        grid[y1][x1] = '↓'
                    elif x1 == x2 and y1 > y2:
                        grid[y1][x1] = '↑'
                    elif y1 == y2 and x1 < x2:
                        grid[y1][x1] = '→'
                    elif y1 == y2 and x1 > x2:
                        grid[y1][x1] = '←'

                x1, y1 = vertices[-1]
                x2, y2 = vertices[0]
                if x1 == x2 and y1 < y2:
                    grid[y1][x1] = '↓'
                elif x1 == x2 and y1 > y2:
                    grid[y1][x1] = '↑'
                elif y1 == y2 and x1 < x2:
                    grid[y1][x1] = '→'
                elif y1 == y2 and x1 > x2:
                    grid[y1][x1] = '←'

            for row in grid:
                outfile.write(' '.join(row) + '\n')
            outfile.write("\n")

        score, distance, lenth, variance = solution.evaluate()
        outfile.write(
            f"Score: {score}, Distance: {distance}, Length: {lenth}, Variance: {variance}\n")
        outfile.write("=" * 40 + "\n")

    # Save ring_vertices and ring_edges to a separate file
    best_imr_file = "./Experiment Data/bestIMR.txt"
    os.makedirs(os.path.dirname(best_imr_file), exist_ok=True)

    with open(best_imr_file, 'w') as best_imr_outfile:
        pass  # clear all

    with open(best_imr_file, 'a') as best_imr_outfile:
        # Save ring_vertices
        best_imr_outfile.write("Ring Vertices:\n")
        for key, vertices in solution.ring_vertices.items():
            vertex_str = ' -> '.join([f"({x},{y})" for x, y in vertices])
            best_imr_outfile.write(f"{key}: {vertex_str}\n")

        # Save edge_count
        best_imr_outfile.write("Edge Count:\n")
        for edge, count in solution.edge_count.items():
            edge_str = f"(({edge[0][0]},{edge[0][1]}),({edge[1][0]},{edge[1][1]})): {count}"
            best_imr_outfile.write(f"{edge_str}\n")

    print(
        f"Saved generation {generation} to {file_path} and bestIMR data to {best_imr_file}")


def select(population):
    return min(population, key=lambda sol: sol.evaluate()[0])


def construct_routing_table(rings, N, output_file='./Experiment Data/IMR/routing_table.txt'):
    output = {}

    for x1 in range(N):
        for y1 in range(N):
            for x2 in range(N):
                for y2 in range(N):
                    if (x1, y1) == (x2, y2):
                        continue

                    pair_key = ((x1, y1), (x2, y2))
                    output[pair_key] = {}

                    for ring_id, ring in enumerate(rings):
                        if (x1, y1) in ring and (x2, y2) in ring:
                            # distance
                            idx1 = ring.index((x1, y1))
                            idx2 = ring.index((x2, y2))

                            if idx1 <= idx2:
                                distance_ab = idx2 - idx1
                            else:
                                distance_ab = len(ring) - (idx1 - idx2) - 1

                            output[pair_key][ring_id] = distance_ab

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for pair_key, rings_dict in output.items():
            a, b = pair_key
            xa, ya = a
            xb, yb = b
            nodea = xa + N * ya
            nodeb = xb + N * yb
            f.write(f"Pair {(nodea, nodeb)}:\n")
            for ring_id, length in sorted(rings_dict.items(), key=lambda item: item[1]):
                f.write(f"  Ring {ring_id}: Length = {length}\n")
            f.write("\n")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run the IMR solution saver")

    parser.add_argument('--N', type=int, default=4,
                        help='Size of the network (N*N)')
    parser.add_argument('--k', type=int, default=9,
                        help='Number of binary strings (rings) in a solution')
    parser.add_argument('--p', type=float, default=0.2,
                        help='Mutation probability')
    parser.add_argument('--num_solution', type=int, default=32,
                        help='Number of solutions in a population')
    parser.add_argument('--lam', type=float, default=0.01,
                        help='Hyperparameter')
    parser.add_argument('--lam_1', type=float,
                        default=1, help='Hyperparameter')
    parser.add_argument('--file_path', type=str, default='./Experiment Data/IMR/n=4,k=9,o.txt',
                        help='Path to save the solution data')
    parser.add_argument('--gen_num', type=int, default=1000,
                        help='Number of generations')
    parser.add_argument('--objective', type=str,
                        default='default', help='mode of objective function')

    args = parser.parse_args()

    print(f"Running with: N={args.N}, k={args.k}, p={args.p}, num_solution={args.num_solution}, lam={args.lam}, file_path={args.file_path}, gen_num={args.gen_num}")

    population = [Solution(N=args.N, k=args.k, p=args.p, lam=args.lam, lam_1=args.lam_1,
                           objective=args.objective) for _ in range(args.num_solution)]
    final_population = evolve_population(
        args.N, args.k, args.p, args.lam, args.lam_1, args.objective, population, num_solution=args.num_solution, file_path=args.file_path, gen_num=args.gen_num)


def load_best_imr(file_path='./Experiment Data/bestIMR.txt'):
    ring_vertices = {}
    edge_count = {}

    with open(file_path, 'r') as infile:
        lines = infile.readlines()

        i = 0
        while i < len(lines):
            if lines[i].strip() == "Ring Vertices:":
                i += 1
                while i < len(lines) and lines[i].strip() != "Edge Count:":
                    key, vertices = lines[i].split(": ")

                    # 处理并清理字符串，避免空字符串或无效字符
                    vertices_list = []
                    for vertex in vertices.split(' -> '):
                        vertex = vertex.strip('()\n ')
                        if vertex:  # 确保不是空字符串
                            try:
                                vertices_list.append(
                                    tuple(map(int, vertex.split(','))))
                            except ValueError as e:
                                print(
                                    f"Skipping invalid vertex data: {vertex}")

                    ring_vertices[int(key)] = vertices_list
                    i += 1

            elif lines[i].strip() == "Edge Count:":
                i += 1
                while i < len(lines):
                    edge_str, count_str = lines[i].split(": ")
                    edge = tuple(
                        tuple(map(int, point.strip('()\n ').split(',')))
                        for point in edge_str.strip().split('),(')
                    )
                    count = int(count_str)
                    edge_count[edge] = count
                    i += 1

            else:
                i += 1

    return ring_vertices, edge_count


if __name__ == "__main__":
    main()
