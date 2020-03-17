import json

# 获取外包矩形
def get_rect_bounds(points, r = 0):
  length = len(points)
  top = down = left = right = points[0]
  for i in range(1, length):
    if points[i]['x'] > right['x']:
      right = points[i]
    elif points[i]['x'] < left['x']:
      left = points[i]
    else:
      pass
    if points[i]['y'] > down['y']:
      down = points[i]
    elif points[i]['y'] < top['y']:
      top = points[i]
    else:
      pass
  top_left = {'x': left['x'] - r, 'y': top['y'] - r}
  top_right = {'x': right['x'] + r, 'y':top['y'] - r}
  down_right = {'x': right['x'] + r, 'y':down['y'] + r}
  down_left = {'x': left['x'] - r, 'y':down['y'] + r}
  return [top_left, top_right, down_right, down_left]

# 判断点是否在外包矩形内 可以过滤掉大部分点
def is_point_in_rect(point, rectbounds):
    top_left = rectbounds[0]
    top_right = rectbounds[1]
    down_right = rectbounds[2]
    down_left = rectbounds[3]
    return (down_left['x'] <= point['x'] <= top_right['x']
            and top_left['y'] <= point['y'] <= down_right['y'])


# points 默认按顺序 逆时针或顺时针
def is_point_in_polygon(point, points):
    rect_bounds = get_rect_bounds(points)
    if not is_point_in_rect(point, rect_bounds):
        return False
    length = len(points)
    point_start = points[0]
    flag = False
    for i in range(1, length):
        point_end = points[i]
        # 点与多边形顶点重合
        if (point['x'] == point_start['x'] and point['y'] == point_start['y']) or (
                point['x'] == point_end['x'] and point['y'] == point_end['y']):
            return True
        # 判断线段两端点是否在射线两侧
        if (point_end['y'] < point['y'] <= point_start['y']) or (
                point_end['y'] >= point['y'] > point_start['y']):
            # 线段上与射线 Y 坐标相同的点的 X 坐标
            if point_end['y'] == point_start['y']:
                x = (point_start['x'] + point_end['x']) / 2
            else:
                x = point_end['x'] - (point_end['y'] - point['y']) * (
                        point_end['x'] - point_start['x']) / (
                        point_end['y'] - point_start['y'])
            # 点在多边形的边上
            if x == point['x']:
                return True
            # 射线穿过多边形的边界
            if x > point['x']:
                flag = not flag
            else:
                pass
        else:
            pass
 
        point_start = point_end
    return flag

def get_subgraph(points, r):
  name = 'bn-mouse-kasthuri'
  data_path = './data/' + name + '/graph-with-pos.json'
  with open(data_path) as graph_data:
      graph = json.loads(graph_data.read())
      allnodes = graph['nodes'] #allnodes是按id递增的顺序
      alllinks = graph['links']
  points_with_pos = []
  for i in range(0, len(points)):
    points_with_pos.append(allnodes[int(points[i])])
  # print(points_with_pos)

  inner_points = []
  for i in range(0, len(allnodes)):
    if is_point_in_polygon(allnodes[i], points_with_pos):
      inner_points.append(allnodes[i])

  # 对inner_points的每个点 加入半径为R范围内的点
  enlarged_rect_bounds = get_rect_bounds(points_with_pos, r)
  res_points = []
  res_points_id = []
  for i in range(0, len(allnodes)):
    # 先过滤掉绝对不满足的点
    if not is_point_in_rect(allnodes[i], enlarged_rect_bounds):
      pass
    else:
      flag = False
      for j in range(0, len(inner_points)):
        dis = (inner_points[j]['x'] - allnodes[i]['x']) * (inner_points[j]['x'] - allnodes[i]['x']) + (inner_points[j]['y'] - allnodes[i]['y']) * (inner_points[j]['y'] - allnodes[i]['y'])
        if dis < r*r:
          flag = True
          break
      if flag:
        res_points.append(allnodes[i])
        res_points_id.append(allnodes[i]['id'])
  # res_points = list(set(res_points))
  # 获取起点和终点均在res_points里的边
  res_links = []
  for i in range(0, len(alllinks)):
    if alllinks[i]['source'] in res_points_id and alllinks[i]['target'] in res_points_id:
      res_links.append(alllinks[i])
  return {
    'nodes': res_points,
    'links': res_links
  }
