{
    extras: std.parseJson(std.extVar('HPARAMS')),

    get(defaults)::
        local default_hparams = defaults;
        local extras = self.extras;
        local hparams = default_hparams + extras;
        # Make sure that no extra arguments are passed
        assert (std.length(hparams) == std.length(default_hparams));
        hparams,
}
