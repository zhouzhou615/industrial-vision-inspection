package com.vision.inspect.service;

import com.vision.inspect.model.InspectResult;
import org.springframework.stereotype.Component;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 流水线检测结果存储：保存最近若干件的检测结果 + 累计统计，供看板轮询展示。
 */
@Component
public class LineResultStore {

    private static final int MAX_KEEP = 50;

    private final Deque<InspectResult> recent = new ArrayDeque<>();
    private final AtomicLong total = new AtomicLong();
    private final AtomicLong okCount = new AtomicLong();
    private final AtomicLong ngCount = new AtomicLong();
    private volatile long seq = 0;

    public synchronized InspectResult add(InspectResult r) {
        seq++;
        total.incrementAndGet();
        if (r.isPassed()) {
            okCount.incrementAndGet();
        } else {
            ngCount.incrementAndGet();
        }
        recent.addFirst(r);
        while (recent.size() > MAX_KEEP) {
            recent.removeLast();
        }
        return r;
    }

    public synchronized Snapshot snapshot() {
        Snapshot s = new Snapshot();
        s.total = total.get();
        s.ok = okCount.get();
        s.ng = ngCount.get();
        s.seq = seq;
        s.recent = new ArrayList<>(recent);
        return s;
    }

    public synchronized void reset() {
        recent.clear();
        total.set(0);
        okCount.set(0);
        ngCount.set(0);
        seq = 0;
    }

    /** 看板快照 */
    public static class Snapshot {
        public long total;
        public long ok;
        public long ng;
        public long seq;
        public List<InspectResult> recent;
    }
}
