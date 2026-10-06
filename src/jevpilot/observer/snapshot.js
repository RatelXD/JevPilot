({ snapshotId, navigationId, allowedOrigins, allowedOperations, candidateLimit, textLimit }) => {
  const stateKey = "__jevpilotObserverState";
  const previous = globalThis[stateKey];
  const documentId =
    previous?.documentId ??
    `document-${crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`}`;

  const normalizedText = (value) =>
    String(value ?? "")
      .replace(/\s+/g, " ")
      .trim();

  const visibleText = () => {
    if (!document.body) return "";
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
    );
    const range = document.createRange();
    const fragments = [];
    let length = 0;
    let node = walker.nextNode();
    while (node && length < textLimit) {
      const parent = node.parentElement;
      const value = normalizedText(node.textContent);
      if (
        parent &&
        value &&
        !parent.closest("script,style,noscript,template,[hidden],[aria-hidden='true']")
      ) {
        const style = getComputedStyle(parent);
        if (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          style.visibility !== "collapse" &&
          Number(style.opacity) !== 0
        ) {
          range.selectNodeContents(node);
          const rect = range.getBoundingClientRect();
          if (
            rect.width > 0 && rect.height > 0 &&
            rect.right > 0 && rect.left < window.innerWidth &&
            rect.bottom > 0 && rect.top < window.innerHeight
          ) {
            fragments.push(value);
            length += value.length;
          }
        }
      }
      node = walker.nextNode();
    }
    return normalizedText(fragments.join(" ")).slice(0, textLimit);
  };

  const computedRole = (node) => {
    const explicit = normalizedText(node.getAttribute("role")).toLowerCase();
    if (explicit) return explicit;
    const tag = node.tagName.toLowerCase();
    if (tag === "a" && node.hasAttribute("href")) return "link";
    if (tag === "button") return "button";
    if (tag === "summary") return "button";
    if (tag === "input") {
      const type = (node.getAttribute("type") || "text").toLowerCase();
      if (["password", "file", "hidden"].includes(type)) return null;
      if (["button", "submit", "reset", "image"].includes(type)) return "button";
      if (type === "checkbox") return "checkbox";
      if (type === "radio") return "radio";
      if (["search"].includes(type)) return "searchbox";
      if (["text", "email", "url", "tel", "number"].includes(type)) return "textbox";
    }
    if (tag === "textarea" || node.isContentEditable) return "textbox";
    if (tag === "select") return "combobox";
    return "button";
  };

  const accessibleName = (node) => {
    const ariaLabel = normalizedText(node.getAttribute("aria-label"));
    if (ariaLabel) return ariaLabel;
    const labelledBy = normalizedText(node.getAttribute("aria-labelledby"));
    if (labelledBy) {
      const label = labelledBy
        .split(" ")
        .map((id) => document.getElementById(id)?.textContent ?? "")
        .join(" ");
      if (normalizedText(label)) return normalizedText(label);
    }
    if (node.labels?.length) {
      const labels = Array.from(node.labels, (label) => label.textContent ?? "").join(" ");
      if (normalizedText(labels)) return normalizedText(labels);
    }
    for (const attribute of ["alt", "title"]) {
      const value = normalizedText(node.getAttribute(attribute));
      if (value) return value;
    }
    const placeholder = normalizedText(node.getAttribute("placeholder"));
    if (placeholder) return placeholder;
    if (node instanceof HTMLInputElement) {
      const type = (node.type || "text").toLowerCase();
      if (["button", "submit", "reset"].includes(type) && normalizedText(node.value)) {
        return normalizedText(node.value);
      }
    }
    return normalizedText(node.innerText || node.textContent);
  };

  const isVisible = (node) => {
    if (!(node instanceof HTMLElement) || !node.isConnected) return false;
    if (node.hidden || node.closest("[hidden],[aria-hidden='true']")) return false;
    const style = getComputedStyle(node);
    if (
      style.display === "none" ||
      style.visibility === "hidden" ||
      style.visibility === "collapse" ||
      Number(style.opacity) === 0
    ) {
      return false;
    }
    const rect = node.getBoundingClientRect();
    return (
      rect.width > 0 &&
      rect.height > 0 &&
      rect.right > 0 &&
      rect.left < window.innerWidth &&
      rect.bottom > 0 &&
      rect.top < window.innerHeight
    );
  };

  const isEnabled = (node) =>
    !node.matches(":disabled") &&
    node.getAttribute("aria-disabled") !== "true" &&
    !node.closest("[aria-disabled='true']");

  const isReadonly = (node) =>
    node.matches("[readonly]") || node.getAttribute("aria-readonly") === "true";

  const nullableBoolean = (node, property, attribute) => {
    if (property in node && typeof node[property] === "boolean") return node[property];
    if (!node.hasAttribute(attribute)) return null;
    return node.getAttribute(attribute) === "true";
  };

  const currentValue = (node) => {
    if (node instanceof HTMLInputElement) {
      const type = (node.type || "text").toLowerCase();
      if (["password", "file"].includes(type)) return null;
      return node.value;
    }
    if (node instanceof HTMLButtonElement) return node.value || null;
    if (node instanceof HTMLTextAreaElement) return node.value;
    if (node instanceof HTMLSelectElement) return node.value;
    if (node.isContentEditable) return node.innerText;
    return node.getAttribute("aria-valuetext") ?? node.getAttribute("aria-valuenow");
  };

  const resolvedHref = (node) => {
    if (!(node instanceof HTMLAnchorElement) || !node.href) return null;
    return node.href;
  };

  const contextGuard = (node) => {
    const scope = node.closest("form,[role='dialog'],[role='row'],tr");
    if (!scope) return null;
    const fields = Array.from(
      scope.querySelectorAll("input:not([type='password']):not([type='file']),select,textarea")
    ).map((field) => ({
      tag: field.tagName.toLowerCase(),
      type: field.getAttribute("type") ?? "",
      name: field.getAttribute("name") ?? "",
      id: field.id,
      value: "value" in field ? field.value : "",
      checked: "checked" in field ? Boolean(field.checked) : null,
    }));
    return {
      tag: scope.tagName.toLowerCase(),
      role: scope.getAttribute("role") ?? "",
      name: accessibleName(scope),
      text: normalizedText(scope.innerText || scope.textContent),
      fields,
    };
  };

  const semanticGuard = (node) => ({
    role: computedRole(node),
    name: accessibleName(node),
    value: currentValue(node),
    tag: node.tagName.toLowerCase(),
    inputType: node.getAttribute("type") ?? "",
    maxLength: "maxLength" in node && node.maxLength >= 0 ? node.maxLength : null,
    options:
      node instanceof HTMLSelectElement
        ? Array.from(node.options, (option) => ({
            value: option.value,
            label: normalizedText(option.label || option.textContent),
            disabled:
              option.disabled ||
              option.parentElement?.matches("optgroup[disabled]") === true,
          }))
        : null,
    checked: nullableBoolean(node, "checked", "aria-checked"),
    selected: nullableBoolean(node, "selected", "aria-selected"),
    expanded: nullableBoolean(node, "expanded", "aria-expanded"),
    href: resolvedHref(node),
    enabled: isEnabled(node),
    readonly: isReadonly(node),
    context: contextGuard(node),
  });

  const equalGuard = (left, right) => JSON.stringify(left) === JSON.stringify(right);
  const nodes = new Map();
  const guards = new Map();
  const controls = [];
  const omittedCounts = {
    hidden: 0,
    disabled: 0,
    readonly: 0,
    unnamed: 0,
    overLimit: 0,
  };
  const selector = [
    "button",
    "a[href]",
    "summary",
    "input[type='button']",
    "input[type='submit']",
    "input[type='reset']",
    "input[type='image']",
    "input[type='checkbox']",
    "input[type='radio']",
    "input:not([type='password']):not([type='file']):not([type='hidden']):not([type='button']):not([type='submit']):not([type='reset']):not([type='image']):not([type='checkbox']):not([type='radio'])",
    "textarea",
    "select",
    "[contenteditable='true']",
    "[role='button']",
    "[role='link']",
    "[role='checkbox']",
    "[role='radio']",
    "[role='menuitem']",
    "[role='menuitemcheckbox']",
    "[role='menuitemradio']",
    "[onclick]",
  ].join(",");

  const allowed = new Set(allowedOperations);
  const reserveControlCandidates =
    Number(allowed.has("scroll_up") && window.scrollY > 0) +
    Number(
      allowed.has("scroll_down") &&
        window.scrollY + window.innerHeight <
          document.documentElement.scrollHeight - 2,
    ) +
    Number(allowed.has("wait"));
  const domCandidateLimit = Math.max(
    0,
    candidateLimit - reserveControlCandidates,
  );
  let candidateCount = 0;
  {
    for (const node of document.querySelectorAll(selector)) {
      if (!(node instanceof HTMLElement)) continue;
      if (!isVisible(node)) {
        omittedCounts.hidden += 1;
        continue;
      }
      if (!isEnabled(node)) {
        omittedCounts.disabled += 1;
        continue;
      }
      if (isReadonly(node)) {
        omittedCounts.readonly += 1;
        continue;
      }
      const name = accessibleName(node);
      if (!name) {
        omittedCounts.unnamed += 1;
        continue;
      }
      const href = resolvedHref(node);
      if (href && !allowedOrigins.includes(new URL(href, document.baseURI).origin)) {
        continue;
      }
      const isNativeSelect = node instanceof HTMLSelectElement && !node.multiple;
      const isEditable =
        (node instanceof HTMLInputElement &&
          ["text", "search", "email", "url", "tel", "number"].includes(node.type)) ||
        node instanceof HTMLTextAreaElement ||
        node.isContentEditable;
      const clickTarget =
        node instanceof HTMLButtonElement ||
        node instanceof HTMLAnchorElement ||
        node instanceof HTMLElement && node.tagName === "SUMMARY" ||
        node instanceof HTMLInputElement && ["button", "submit", "reset", "image", "checkbox", "radio"].includes(node.type) ||
        node.matches("[role='button'],[role='link'],[role='checkbox'],[role='radio'],[role='menuitem'],[role='menuitemcheckbox'],[role='menuitemradio'],[onclick]");
      const operations = [];
      if (allowed.has("click") && clickTarget) operations.push("click");
      if (allowed.has("type_text") && isEditable) operations.push("type_text");
      const options = isNativeSelect
        ? Array.from(node.options, (option, index) => ({
            optionRef: String(index),
            name: accessibleName(option),
            value: option.value,
            selected: option.selected,
            disabled: option.disabled || option.parentElement?.matches("optgroup[disabled]") === true,
          })).filter((option) => !option.disabled)
        : [];
      if (allowed.has("select") && isNativeSelect && options.length > 0) operations.push("select");
      const addedCandidates = operations.reduce(
        (count, operation) => count + (operation === "select" ? options.length : 1),
        0,
      );
      if (addedCandidates === 0) continue;
      if (candidateCount + addedCandidates > domCandidateLimit) {
        omittedCounts.overLimit += 1;
        continue;
      }
      const targetRef = `target-${controls.length + 1}`;
      const guard = semanticGuard(node);
      nodes.set(targetRef, node);
      guards.set(node, guard);
      candidateCount += addedCandidates;
      controls.push({
        targetRef,
        role: guard.role,
        name: guard.name,
        value: guard.value,
        enabled: guard.enabled,
        readonly: guard.readonly,
        checked: guard.checked,
        selected: guard.selected,
        expanded: guard.expanded,
        href: guard.href,
        maxLength: guard.maxLength,
        operations,
        options,
      });
    }
  }

  const preflight = (node, request) => {
    const { allowedOrigins: currentAllowedOrigins, operation, optionRef } = request;
    if (!(node instanceof HTMLElement) || !node.isConnected) {
      return { ok: false, reason: "node_disconnected" };
    }
    if (node.ownerDocument !== document) {
      return { ok: false, reason: "document_changed" };
    }
    const expected = guards.get(node);
    if (!expected) {
      return { ok: false, reason: "node_not_in_registry" };
    }
    const href = resolvedHref(node);
    if (href && !currentAllowedOrigins.includes(new URL(href, document.baseURI).origin)) {
      return { ok: false, reason: "href_origin_not_allowed" };
    }
    if (!equalGuard(expected, semanticGuard(node))) {
      return { ok: false, reason: "semantic_guard_changed" };
    }
    if (operation === "type_text") {
      const editable =
        (node instanceof HTMLInputElement &&
          ["text", "search", "email", "url", "tel", "number"].includes(node.type)) ||
        node instanceof HTMLTextAreaElement ||
        node.isContentEditable;
      if (!editable || isReadonly(node)) {
        return { ok: false, reason: "field_not_editable" };
      }
    }
    if (operation === "select") {
      if (!(node instanceof HTMLSelectElement) || node.multiple) {
        return { ok: false, reason: "select_not_supported" };
      }
      const optionIndex = Number(optionRef);
      const option = Number.isInteger(optionIndex) ? node.options[optionIndex] : null;
      if (
        !option ||
        option.disabled ||
        option.parentElement?.matches("optgroup[disabled]") === true
      ) {
        return { ok: false, reason: "option_not_available" };
      }
    }
    if (!isVisible(node)) {
      return { ok: false, reason: "node_not_visible" };
    }
    const rect = node.getBoundingClientRect();
    const x = Math.max(0, Math.min(innerWidth - 1, rect.left + rect.width / 2));
    const y = Math.max(0, Math.min(innerHeight - 1, rect.top + rect.height / 2));
    const hit = document.elementFromPoint(x, y);
    if (!hit || (hit !== node && !node.contains(hit))) {
      return { ok: false, reason: "target_occluded" };
    }
    return { ok: true };
  };

  globalThis[stateKey] = {
    documentId,
    navigationId,
    snapshotId,
    nodes,
    guards,
    preflight,
  };

  return {
    documentId,
    navigationId,
    snapshotId,
    url: location.href,
    title: document.title,
    text: visibleText(),
    scrollY: Math.max(0, Math.trunc(window.scrollY)),
    scrollHeight: Math.max(1, Math.trunc(document.documentElement.scrollHeight)),
    viewportHeight: Math.max(1, Math.trunc(window.innerHeight)),
    waitState: {
      url: location.href,
      title: document.title,
      text: normalizedText(document.body?.innerText ?? ""),
    },
    controls,
    omittedCounts,
  };
}
