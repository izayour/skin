// Service worker for one job only: showing the "next photo is due" reminder.
//
// On Android, `new Notification(...)` throws from a page -- notifications have
// to come from a registration's showNotification, which needs a worker to
// exist. That is the whole reason this file is here.
//
// It deliberately has NO fetch handler. A worker that caches requests is
// exactly how this app's pages went stale before (which is why the local
// server sends no-store and every script carries a ?v= build number), and a
// stale index.html is far more damaging than a missed notification. With no
// fetch handler the browser goes to the network as it always did.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", e => e.waitUntil(self.clients.claim()));

// Tapping the reminder should land on the app, not on a second copy of it:
// focus a window that is already open, and only open one if there is none.
self.addEventListener("notificationclick", e => {
  e.notification.close();
  e.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then(list => {
      for (const c of list) if ("focus" in c) return c.focus();
      if (self.clients.openWindow) return self.clients.openWindow("./");
    })
  );
});
