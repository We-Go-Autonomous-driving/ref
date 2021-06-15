def drive(bbox, cx, left_limit, left_max, right_limit, right_max, turn, frame, stable_min_dist, person_distance, stable_max_dist,speed,max_speed,min_speed,start_speed_down=200):
    # Target의 위치 파악(좌우 회전이 필요한 지)
    if cx <= left_limit: 
        key = 'turn_left' # bbox 중앙점 x좌푯값이 좌측 회전 한곗값(left_limit)보다 작으면 좌회전
        # if bbox[0] < 0 or cx < left_max: key = 'angular_speed_up'
        # elif turn > 0.8: turn = 0.8
        # turn = 0.2
        # speed = 0.5
    elif cx >= right_limit: 
        key = 'turn_right' # bbox 중앙점 x좌푯값이 우측 회전 한곗값(right_limit)보다 크면 우회전
        # if bbox[2] >= frame.shape[1] or cx > right_max: key = 'augular_speed_up'
        # elif turn > 0.8: turn = 0.8
        # turn = 0.2
        # speed = 0.5
    else: # 좌/우 회전이 아니라면 직진, 거리에 따른 속도 제어
        if stable_min_dist <= person_distance <= stable_max_dist:
            key = 'go' # 1.5 ~ 2.0m라면 전진
            speed = 0.5
        else: # 2.0m 초과라면 거리에 따른 속도 증감
            remaining_distance = person_distance - stable_max_dist # 안정 거리 최대값과 사람과의 거리의 차이, 이를 이용해 얼만큼 속도를 증감해야하는 지 정하는 요소
            """
            기본 컨셉은 remaining_distance 200 이상이면 속도 증가, 미만이면 속도 감소(대신 속도 최대값은 2.5, 최소값은 1.2로 설정한다)
            예를 들어 사람과 로봇의 거리가 2500이라면 remaining_distance 500이 된다. 로봇은 speed_fremaining_distanceactor이 200이 될 때까지 속도 증가(최댓값은 2.5로 설정함.)
            remaining_distance 200 미만이 되는 순간 속도 감소(최솟값은 1.2로 설정)
            그리고 remaining_distance 0이 되면 안정 구간 진입이므로 속도는 1로 설정됨.
            
            <아래는 로봇과 사람이 2.5m 떨어진 상황>
            ┌─┐
            └─┘<----위험(stop)---->|<---안정(go)--->|<---속도 증가(최댓값: 2.5)--->|<---속도 감소(최솟값: 1.2)--->|
            0  0                    1.5m            2.0m                          2.3m                          2.5m
            """
            if remaining_distance >= start_speed_down: # speed up
                if speed < max_speed:
                        key = 'linear_speed_up'
                else:
                    speed = max_speed
            else: # speed down
                if speed > min_speed:
                    key = 'linear_speed_down'
                else:
                    speed = min_speed
    return key