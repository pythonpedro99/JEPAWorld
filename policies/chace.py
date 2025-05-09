def waypoints_to_actions(
        self, degree: int = 3, smoothing: float = 0.0, oversample: int = 1000
    ) -> None:
        """
        Robustly fit & re-sample a B-spline through self.waypoints,
        then convert to discrete actions. Returns (actions, smoothed_curve).
        """
        # 1) Dedupe exact repeats
        pts = []
        for p in self.waypoints:
            if not pts or p != pts[-1]:
                pts.append(p)
        if len(pts) < 2:
            return None

        xs, ys = zip(*pts)

        # 2) Try splprep with decreasing k
        curve = None
        for k in range(min(degree, len(xs) - 1), 0, -1):
            try:
                tck, _ = splprep([xs, ys], k=k, s=smoothing)
                # oversample + arc-length re-sample
                u_fine = np.linspace(0.0, 1.0, oversample)
                xf, yf = splev(u_fine, tck)
                coords = np.stack([xf, yf], axis=1)

                deltas = np.diff(coords, axis=0)
                seg_lens = np.linalg.norm(deltas, axis=1)
                cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
                total_L = cumlen[-1]
                n_steps = int(np.floor(total_L / self.forward_step))
                target_L = np.linspace(0.0, n_steps * self.forward_step, n_steps + 1)

                ui = np.interp(target_L, cumlen, u_fine)
                xs_s, ys_s = splev(ui, tck)
                curve = list(zip(xs_s, ys_s))
                break
            except ValueError:
                # try with k-1
                continue

        # 3) If we never got a spline, fall back to linear re-sample
        if curve is None:
            # Just linearly walk and re-sample at forward_step
            # build cumulative length on pts:
            coords = np.array(pts)
            deltas = np.diff(coords, axis=0)
            seg_lens = np.linalg.norm(deltas, axis=1)
            cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
            total_L = cumlen[-1]
            n_steps = int(np.floor(total_L / self.forward_step))
            target_L = np.linspace(0.0, n_steps * self.forward_step, n_steps + 1)
            # find segment for each target_L
            curve = []
            for L in target_L:
                idx = np.searchsorted(cumlen, L) - 1
                idx = max(0, min(idx, len(pts) - 2))
                t = (L - cumlen[idx]) / (cumlen[idx + 1] - cumlen[idx] + 1e-8)
                p0, p1 = np.array(pts[idx]), np.array(pts[idx + 1])
                curve.append(tuple((1 - t) * p0 + t * p1))

        # 4) Discretize into actions
        actions: List[int] = []
        yaw = self.agent_yaw
        pos = np.array(curve[0])

        def norm(a):
            return ((a + 180) % 360) - 180

        for pt in curve[1:]:
            tgt = np.array(pt)
            delta = tgt - pos
            dist = np.linalg.norm(delta)
            if dist < self.forward_step / 2:
                continue

            # turn
            desired = np.degrees(np.arctan2(delta[1], delta[0]))
            dy = norm(desired - yaw)
            n_turns = int(round(abs(dy) / self.turn_step))
            turn_cmd = 0 if dy > 0 else 1
            actions += [turn_cmd] * n_turns
            yaw = norm(yaw + np.sign(dy) * n_turns * self.turn_step)

            # forward
            n_fwd = int(round(dist / self.forward_step))
            actions += [2] * n_fwd

            pos = tgt

        # store & return
        self.actions.append(actions)
        self.waypoints_smoothed.append(curve)