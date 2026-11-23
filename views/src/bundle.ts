// The bundle's entry: what this project contributes to the viewer. `fxtr view` builds it into
// dist/ and serves it, rebuilding it as it changes (`fxtr views build` builds it without the
// viewer); each renderer is for a slot, an entity type (or a step or workflow's name), and has an
// id unique in the viewer.

import { EntityInlineLink, EntityPanel, InvocationPanel, renderer, type ViewBundle } from "fxtr-view";
import { behaviorsRenderers } from "fxtr-view-behaviors";
import { AUDIT, AUDIT_WORKFLOW, AuditInlineLink, AuditOverview, AuditPanel } from "./audit";
import { RUN_READER, RunReader } from "./reader";

const bundle: ViewBundle = {
  renderers: [
    // The readable run view comes first, so it is an agent run's default; behaviors' conversation
    // view stays one click away in the pane's view switcher.
    renderer(RUN_READER, RunReader),
    ...behaviorsRenderers,
    renderer(EntityPanel, AUDIT, "malt-fxtr.AuditPanel", AuditPanel),
    renderer(EntityInlineLink, AUDIT, "malt-fxtr.AuditInlineLink", AuditInlineLink),
    renderer(InvocationPanel, AUDIT_WORKFLOW, "malt-fxtr.AuditOverview", AuditOverview),
  ],
};

export default bundle;
